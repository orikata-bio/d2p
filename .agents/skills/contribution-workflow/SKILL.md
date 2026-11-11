---
name: contribution-workflow
description: Prepare a feature-branch pull request with honest human, agent, source and validation disclosure, preserving required checks and maintainer approval.
---

Read AGENTS.md, CONTRIBUTING.md and the repository PR template. Use a focused
branch; preserve other workers' changes. Check git status and the target SHA.
Run affected checks before submission and preserve required CI. Never direct-push
main, fabricate validation or attribution, bypass licensing/source restrictions,
grant access, or merge external changes without the owner's approval.

Fill every required PR section with concrete facts: Summary, Human contribution,
Agent contribution, External sources, and Validation. Identify human decisions
and verification separately from agent output; include agent/model versions when
known, scope of autonomy, and any limitations. List external code URLs, revisions,
licenses, affected paths, and how each source was used. Explicit "None" is valid
where true. Do not include secrets or private transcripts.

Public and release repositories do not use Greptile or CodeRabbit reviews.
In private -dev repositories, configured vendor reviews apply only to PRs into
main. Read existing feedback using the review-feedback skill when available;
never infer permission for a new review, extra reviewer, or repeated review loop.
Before promotion, bind the accepted dev commit and public source PR/revision.
Use the established release export/allowlist and a separate release PR; preserve
its required checks and human review. Do not export private history or notes.
