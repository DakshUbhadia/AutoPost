# AutoPost

AutoPost is a local-first reference implementation of an auditable post-training control plane for verifiable mathematical rewards.

Sprint 0 establishes the reproducible development and API-validation baseline. Small-model work validates engineering behavior only; it is not evidence of target-model scientific improvement.

## CPU development

Use CPython 3.13.7 in a fresh virtual environment:

```text
make install-cpu
make test
make selfcheck
```

The CPU environment does not download models or install the GPU training stack.

## Toolchain API probe

Create and activate a separate virtual environment, then run:

```text
make install-toolchain-probe
make toolchain-probe
```

The checked-in probe lock reproduces the validated CPython 3.13.7 Windows CPU inspection environment. It validates the pinned TRL/Transformers/PEFT API surface but does not claim that vLLM or a GPU training path has run; those runtime checks remain hardware-gated.

## Design documents

- [Architecture and scientific protocol](post_training_documentation.md)
- [Implementation sprint roadmap](implementation_sprint_roadmap.md)
