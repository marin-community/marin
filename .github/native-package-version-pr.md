Advance Marin-owned native package compatibility floors and the universal lock
to published releases. Native source changes remain source-built in their
implementation pull requests; this dependency pull request exercises the
published wheels before they become the repository default.

The updater merges this pull request when its checks pass. If the update changes
environment markers in `uv.lock`, a maintainer reviews those changes and merges it.

Part of #6456
