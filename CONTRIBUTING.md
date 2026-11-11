# Contributing

<!-- BEGIN REPOSITORY-CONTRIBUTION-WORKFLOW-V1 -->
## Public contribution workflow

Public pull requests are welcome. Use a fork and a focused branch; target `main`.
Submit the required human, agent, source, and validation disclosure below. No
Kaname launcher, recording, transcript, enforcement marker, or provenance task is
required for public contributions. Preserve historical evidence without claiming
that uncaptured work was recorded. This does not change separately scoped private
development requirements.

Greptile and CodeRabbit do not review this public repository. Maintainer review
and applicable CI remain required. A maintainer-approved external PR may be
ported to its explicitly paired private dev repo, tested there, and returned
through a new release PR. The dev PR and release PR must remain linked to the
original public PR and its exact reviewed commit. Conflicts or different release
layouts require an explicit port; never replace a release tree with a dev tree.
Cross-repository automation must be configured and authenticated before it is
described as active. It may open PRs but may not merge release PRs automatically.

## Required contribution disclosure

Every pull request, including maintenance and release PRs, must state:

- **Human contribution:** who directed the work, wrote or changed code, reviewed
  it, made design decisions, and ran validation. Identify the accountable human.
- **Agent contribution:** each agent/tool and model/version when known; what it
  generated, changed, researched, reviewed, or executed; the scope of autonomy;
  and how the human checked its output. Say "None" if no agents were used.
- **External sources:** URLs and exact revisions/versions when available for
  copied, adapted, translated, generated-from, or consulted code; affected files;
  licenses/notices; dependencies, papers, documentation, models, and data. Explain
  what was reused versus independently implemented. Say "None" if applicable.
- **Validation:** exact commands, results, platforms, and meaningful gaps. Do not
  invent test results, captures, authorship, reviews, citations, or permissions.

Do not publish credentials, private chat transcripts, hidden reasoning, customer
data, or private repository details. A clear factual account is required; raw
agent session logs are not required. Preserve existing licenses and notices.
Repository-specific source-use and clean-room requirements still apply.

Only a maintainer can grant organization write access. Outside contributors use
forks and pull requests and must obtain approval before their changes enter an
organization branch or privileged workflow. Approval is tied to the exact head
commit; subsequent changes require approval again. Never push directly to main,
force-push protected branches, or treat a bot review as human approval.
<!-- END REPOSITORY-CONTRIBUTION-WORKFLOW-V1 -->
