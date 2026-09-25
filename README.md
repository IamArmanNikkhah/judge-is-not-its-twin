# The Judge Is Not Its Twin

Code and data for *The Judge Is Not Its Twin: post-training makes a model's writing more predictable but barely moves its taste, as a judge, toward predictable writing* (Arman Nik Khah, 2026; arXiv link to follow).

The paper follows two open model families, OLMo-2 and Zephyr (7B), through their public training stages. Every stage writes short stories, and every stage judges pairs of stories written by the other family. Training makes the writing more predictable. It barely moves the judges' preference for predictable stories, and it breaks "which is more creative?" as a question for trained judges.

## What is here

| Path | Contents |
|---|---|
| `data/items/` | The 1,389 stories from the main study (`bank-pilot.jsonl`) and the ten prompts. |
| `data/runs/` | The raw generation records for those stories, one file per training stage. |
| `data/typicality/` | Surprise scores (mean negative log-likelihood per token) for every story under each reader model. The `-prefix` files are the ones the paper uses (writing prompt in front of the story). |
| `data/validity/` | The competence check: story-versus-ruined-copy pairs and every judge's answers, under "creative", "better" and the grammatical "better" rerun. |
| `data/judge/` | The main measurement: 200 story pairs per judge stage, under "creative" and "better". |
| `data/analysis/` | Every analysis output the paper cites, including the post hoc drift bound and its written rule, and the follow-up (Section 8): screens, pairings, blind-read sheets with their keys, and the break marks for Appendix A. |
| `code/` | Generation, scoring, judging and analysis scripts. |
| `prereg/` | Public copies of the dated design documents in which the rules, thresholds and expected outcomes were written before each run. |
| `configs/` | Model checkpoints per family and the judging protocol. |

## Reproducing the paper's numbers

The analysis needs only numpy and runs in seconds on a laptop:

```bash
pip install numpy
python code/judge_trajectory.py                                # Table 2, all pairs
python code/judge_trajectory.py --min-chars 400 --stage-control  # the stage-difference control (Section 6)
python code/judge_trajectory.py --reader judge-base            # surprise scored by the judge's own family
python code/drift_bound.py item-base                           # drift bound, Section 6
python code/drift_bound.py judge-base                          # drift bound, second reader
python code/gate_report.py                                     # competence check under "creative", Table 3
python code/gate_report.py --glob "data/validity/gate-better-*.jsonl"      # same, under "better"
python code/gate_report.py --glob "data/validity/gate-bettergram-*.jsonl"  # grammatical "better" rerun
python code/pilot_analysis.py --typ-suffix=-prefix            # writer drift, Table 1
```

Each command's output is saved under `data/analysis/` for comparison. Generating stories, scoring them and running judges needs a GPU and the checkpoints listed in `configs/lineages.yaml`; the reported runs used one GPU per job.

Some code comments refer to working notes that are not part of this repository. The paper is the reference for every claim.

## License

Code: MIT (`LICENSE`). Data and the documents in `prereg/`: CC BY 4.0 (`LICENSE-DATA`).
