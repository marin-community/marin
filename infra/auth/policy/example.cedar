// Standing application policy. Runtime grants impose additional ceilings.
permit (
    principal in Marin::Group::"research",
    action == Marin::Action::"iris.job.submit",
    resource in Marin::Namespace::"/analysis/"
);

permit (
    principal in Marin::Group::"incident-response",
    action == Marin::Action::"loom.session.create",
    resource == Marin::LoomProfile::"/diagnostics/"
);

// This authorizes the complete audited native identity, not arbitrary objects.
permit (
    principal in Marin::Group::"incident-response",
    action == Marin::Action::"cw_s3.credentials.issue",
    resource == Marin::CWIdentity::"diagnostics"
) when { resource.read_only };
