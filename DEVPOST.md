# Devpost submission draft

Fill in the bracketed parts after your real runs. Use your actual results, not these placeholders.

---

## Project name
Repro: an agent that checks if ML papers hold up

## Tagline
Give it a paper. It picks a claim, re-runs the experiment itself, and tells you if the number holds up.

## What I built
Repro is an autonomous research engineer. You give it an arXiv id or a PDF. It reads the paper, chooses one quantitative claim that is feasible on your machine (for example "XGBoost reaches RMSE X on California Housing, Table N"), finds the dataset, writes the experiment, runs it, debugs its own errors, and ends with one of four verdicts: **Reproduced, Partially reproduced, Not reproduced, or Could not attempt**. Every run produces a report with the paper's number next to the measured one, a chart, every deviation from the paper's setup, and all the scripts so a human can re-run them.

## Why it is agentic
Repro is not a chatbot that summarizes papers. It follows the agent loop (goal, decide, use tool, observe, finish) for 10 to 40 steps with no human input:

- **Planning:** it checks the machine's hardware first, then picks a claim it can actually test, records it, and writes a step plan that it updates as it goes.
- **Tool use:** 10 tools: arXiv fetch and in-paper search, dataset search across a built-in catalog, OpenML and Hugging Face, a sandboxed Python runner, an allowlisted package installer, and long-term memory.
- **Retrieval before acting:** it searches the paper for the exact reported number, split and hyperparameters before writing any code.
- **Adapting when something fails:** it reads tracebacks and fixes root causes, installs missing packages, and shrinks the data when a run times out. After three failures in a row it is forced to stop and reflect before trying again.
- **Memory:** lessons from failures are saved and injected into every future run, so the agent gets better over time.
- **Honesty guardrails:** the agent cannot report a number its code never printed (every final value is checked against real run output), and the verdict is computed by plain code with fixed tolerances, not decided by the LLM.

## Tools I used
Python, an LLM with tool calling ([model you used]), arXiv API, pypdf, scikit-learn, OpenML and Hugging Face APIs, matplotlib, Streamlit. The LLM layer is provider-agnostic (Anthropic, OpenAI, Gemini, Groq, OpenRouter, Ollama) and uses no SDKs.

## Why I made it
Many ML results are hard to reproduce, and checking one by hand takes hours: finding the number, the data, the split, the hyperparameters, then debugging. As a student I constantly read papers and wonder whether a headline number would survive a re-run. Reproduction is a perfect job for an agent: it is multi-step, tool-heavy, full of failures to recover from, and the final answer can be verified with numbers instead of trusted on vibes.

## Results
| Paper | Claim tested | Paper | Repro | Verdict |
|---|---|---|---|---|
| [paper 1] | [claim] | [x] | [y] | [verdict] |
| [paper 2] | [claim] | [x] | [y] | [verdict] |
| [paper 3, GPU-heavy] | [claim] | [x] | - | Could not attempt: [reason] |

## Challenges
- Papers rarely state every detail, so the agent had to log each assumption as a deviation instead of silently guessing.
- LLMs like to "round up" results. I solved this by checking every reported value against real program output and moving the verdict into deterministic code.
- Long runs overflow the context window, so older tool outputs are trimmed while recent ones stay complete.

## What's next
Multi-claim reports per paper, multiple seeds with confidence intervals by default, Docker-isolated execution, and a public leaderboard of reproduced and non-reproduced claims.

## How to run
See the README in the repo: `pip install -r requirements.txt`, add an API key to `.env`, then `streamlit run app.py`. The offline example (`examples/toy_paper.txt`) works for a quick check.

---

# Demo video script (target 2:30 to 3:00)

**0:00 to 0:20, the problem.** Face cam or title slide. "ML papers report numbers. Do they hold up? Checking one takes hours. I built an agent that does it on its own."

**0:20 to 1:40, live run (use Replay at 4x).**
1. Paste the arXiv id, add a focus hint, press Reproduce. Point at the loop bar at the top: Goal, Decide, Use tool, Observe, Finish.
2. Paper card appears. "It read the paper and found the results table."
3. Claim card appears. "It committed to one number it can actually test on my laptop."
4. Plan card fills in.
5. A code run fails (red). "Here's the important part: it reads its own error and fixes it." Next run turns green.
6. Verdict card and chart. Read the two numbers out loud.

**1:40 to 2:10, honesty features.** Show the full report: the deviations list, then the transcript moment where a finish call was rejected (if it happened), or explain: "It can't report a number its code never printed, and the verdict is computed by code, not by the model."

**2:10 to 2:30, the blocked case.** Show a GPU-heavy paper ending in "Could not attempt" with a specific reason. "An honest no beats a fake yes."

**2:30 to 2:50, memory and close.** Open the Memory tab: "Lessons from failures carry over to future runs." End with the one-line tagline.

# Submission checklist
- [ ] Public GitHub repo with README (no `.env` committed)
- [ ] 2 to 3 real runs done, `report.html` files saved for screenshots
- [ ] Results table above filled in with real numbers
- [ ] Video uploaded (YouTube unlisted works) and linked
- [ ] Screenshots: verdict card, live loop view, report page
- [ ] Submitted before Monday 12 Oct, 07:00 (GMT+3)
