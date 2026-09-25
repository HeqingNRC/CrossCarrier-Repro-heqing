"""Reuse the supplied forward/loss block verbatim via Python's AST.

The original trainer contains unrelated 77-GHz checkpoint-selection and resume
logic. We reuse its math without running that controller. The audit hashes are
checked by load_project before this function is constructed.
"""
from __future__ import annotations
import ast, copy
from pathlib import Path

ARGS = ('model miro domain_clf carrier_adv grl_band_log_ghz grl_bin_edges grl_discrete '
        'grl_bins source_loader x_src y_src d_src x_aux r_src global_step total_steps '
        'logit_prior ce ce_domain').split()


def build_objective(original_module):
    path=Path(original_module.__file__)
    tree=ast.parse(path.read_text(encoding='utf-8'))
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    candidates=[n for n in ast.walk(train) if isinstance(n,ast.With)
                and 'torch.autocast' in ast.unparse(n.items[0].context_expr)]
    if len(candidates)!=1: raise RuntimeError('Cannot uniquely locate the audited autocast loss block.')
    block=candidates[0]
    if not any(isinstance(n,ast.Name) and n.id=='loss_source' for n in ast.walk(block)):
        raise RuntimeError('Audited loss block is missing loss_source.')
    prefix=ast.parse('''
p = global_step / max(1, total_steps)
dann_lambda = float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0) if config.USE_DANN else 0.0
grl_lambda_last = 0.0
v18_lambda_last = 0.0
''').body
    suffix=ast.parse('''
return loss_source, {
    "ce": loss_ce, "supcon": loss_sc, "anchor_named_miro": loss_mi,
    "residual_weighted": loss_v13, "residual_raw": loss_v13_freq,
    "decorrelation_raw": loss_v13_decorr, "grl_raw": loss_grl,
    "dann_raw": loss_dn, "v15": loss_v15, "v15r": loss_v15r,
    "grl_lambda": grl_lambda_last,
}, z_neck_src, logits_adj, raw_logits_src
''').body
    fn=ast.FunctionDef(name='audited_source_objective',
        args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=a) for a in ARGS],kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=prefix+copy.deepcopy(block.body)+suffix,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    namespace=dict(vars(original_module))
    exec(compile(module, str(path)+'::audited_source_objective', 'exec'), namespace)
    return namespace[fn.name], ast.unparse(module)+'\n'
