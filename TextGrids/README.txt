# TextGrids for Child Language Data

This folder contains manually corrected `TextGrid` files for Dutch child speech, organized by dataset and further divided by **age** and **diagnosis** (typically developing children vs. children with TOS, i.e., developmental language disorder).

## Structure

- One folder per dataset:
  - `auris/`
  - `childes/`
- Within each dataset, files are organized into subfolders based on the **age group** and **diagnostic category** of the children.

Each `TextGrid` file corresponds to a `.wav` audio recording. These annotations are meant to support research on child speech and language development, particularly in the context of early screening for speech and language disorders.

## Annotation Process

The `TextGrid` files were initially generated using [Pyannote](https://github.com/pyannote/pyannote-audio), an automatic speaker diarization tool. However, Pyannote’s diarization is not always accurate with regard to speaker identity. To improve the quality of these annotations:

- We manually corrected the TextGrids using [Praat](https://www.fon.hum.uva.nl/praat/).
- Corrections were made by researchers and student annotators, following guidelines (see below).

## Annotation Guidelines

Each `.TextGrid` file includes speaker annotations with the following labels:

| Label   | Meaning                                          |
|---------|--------------------------------------------------|
| `chi`   | Child speech                                     |
| `chi_ns` | Child non-speech (e.g., mumbles, screams, one word utterance such as 'ja', 'nee', 'kijk)       |
| `adu`   | Adult speech                                     |
| `adu1`, `adu2` | Additional adult speakers (if applicable)|
| `noise` | ~2 seconds of background noise (only once/file) |

### Key Instructions Followed During Annotation

1. Open the `.wav` file together with the corresponding `.TextGrid` in Praat.
2. Review and relabel the speaker intervals:
   - `SPEAKER00` / `SPEAKER01` → `chi`, `adu`, etc.
   - Carefully listen and adjust boundaries if needed.
3. Do **not annotate overlapping speech** (child and adult speaking simultaneously).
4. Long pauses **within** child utterances were annotated as part of `chi` to preserve developmental cues.
5. Annotators were instructed to:
   - Add a `noise` label for at least 1.5–2 seconds of background noise.
6. Overlapping speech segments remain in the files as produced by Pyannote. We did **not** remove these because:
   - We use algorithms to filter them out automatically in the screener development project.
   - They may still be useful for other research purposes.

## Notes

- These files were prepared for a project on automatic screening for speech and language disorders in children.
- If you plan to use these annotations in your own research or tools, we recommend reviewing the guidelines above or contacting us for clarification.
- For questions or collaboration, please get in touch!

## Contact

Alex Elio Stasica (eliostasica@gmail.com)
Charlotte Pouw (c.m.pouw@uva.nl)
Louis Berard (berardlouis@gmail.com)

