# Dataset Sources and Provenance (00_data)

This directory holds the `known_people_unknown_freq` task data used for training and evaluation.
The data is a combination of two sources that were **already merged when it was obtained**. This
file documents the composition, the supplementary NRC data, how it is used in training, and an
honest caveat about source separability.

## Primary source — University of Alabama CI4R cross-frequency dataset
A multi-frequency radar micro-Doppler human activity recognition dataset from the Laboratory of
Computational Intelligence for Radar (CI4R), University of Alabama.
- Three **co-located** radars: XeThru (10 GHz UWB impulse), Ancortek SDR (24 GHz FMCW, 1500 MHz
  bandwidth), and Texas Instruments IWR1443 BOOST (77 GHz FMCW, 4 GHz bandwidth). The same
  activity instance is recorded simultaneously by all three radars.
- 11 activities / ambulatory gaits. This task keeps 7 classes for the reported protocol
  (Away, Bend, Kneel, Pick, SStep, Sit, Towards; Crawl, Limp, Scissor, Toes are dropped).
- In this curated task the data appears under 5 subject IDs: `ua_0000, ua_0010, ua_0014,
  ua_0029, ua_0030`.
- Public release: https://github.com/ci4r/CI4R-Activity-Recognition-datasets

## Supplementary source — NRC Canada 24 GHz enrichment ("NRC supplementary data")
The dataset is enriched with the data collected in National Research Council Canada. An 24GHz
FMCW radar of model PUP_EN24C_T2R4 developed by Luswave was used for the data collection process.
A pair of horn antennas, each with a gain of 15dBi were used as the Transmitting (Tx) and
Receiving (Rx) antennas. The radar was placed on a table 1m from the ground. Two volunteers
consented to participate in this study. When recording the data, a volunteer were asked to
perform the activities 3m from the radar at the boresight of the radar line of sight.

This NRC data is **24 GHz only** and enriches the 24 GHz channel of the dataset.

## Training usage (important)
During training the NRC 24 GHz supplement is **used together with the Alabama data**. The source
training pool is the combined **10 GHz + 24 GHz** set (Alabama 10/24 GHz + NRC 24 GHz); the model
is tested **zero-shot on the held-out 77 GHz** (Alabama) data. The NRC supplement is therefore
part of the 24 GHz source domain, not a separate evaluation set.

## Provenance caveat — the NRC subset is NOT separable at the per-file level
The two sources were already merged when the data was obtained, and the merged data carries **no
per-file source label**:
- the manifests have no source column (only `path, frequency, class, class_idx, subject,
  source_file, original_path, original_split`);
- all samples use unified `ua_####` subject IDs (no NRC-specific subject);
- the filename prefix encodes only `[sensor][activity][subject]` (sensor 06=10GHz, 01=24GHz,
  04=77GHz; the middle two digits are the activity index), not the acquisition source;
- this curated `known_people_unknown_freq` task retains the **co-located** captures (each 24 GHz
  sample has a 10/77 GHz counterpart), so standalone NRC-only 24 GHz samples are not individually
  present or marked here (a forensic check found ~0 separable 24 GHz-only samples).

Consequently the NRC supplement **cannot be physically isolated into its own folder from this
package alone**; it is documented here for transparency and provenance. Note that both 24 GHz
contributions (Alabama Ancortek and NRC Luswave PUP_EN24C_T2R4) share the single `24GHz`
frequency label despite being different radar hardware.

Source (primary dataset): https://github.com/ci4r/CI4R-Activity-Recognition-datasets
