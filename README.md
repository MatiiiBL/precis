# Précis

An agentic pipeline that reads and summarizes thousands of research papers in
parallel with LLMs: each paper is read by an agent that navigates its
sections with tools, every claim in the summary is checked against the paper
by a second agent, and the corpus is mapped into topics with a written report
and a browsable site.

Work in progress. The design is in [docs/design.md](docs/design.md).

```sh
make install   # uv sync
make check     # lint, type-check, tests
```
