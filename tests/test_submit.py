from network_glm.submit import lev1 as submit_lev1


def test_expand_all_tasks():
    from network_glm.task_config.loader import get_all_tasks

    assert submit_lev1.resolve_tasks(all_tasks=True, base=False, dual=False, tasks=None) == get_all_tasks()


def test_expand_explicit():
    assert submit_lev1.resolve_tasks(all_tasks=False, base=False, dual=False,
                                     tasks=["flanker", "stroop"]) == ["flanker", "stroop"]


def test_expand_base_and_dual():
    from network_glm.task_config.loader import get_base_tasks, get_dual_tasks

    assert submit_lev1.resolve_tasks(all_tasks=False, base=True, dual=False, tasks=None) == get_base_tasks()
    assert submit_lev1.resolve_tasks(all_tasks=False, base=False, dual=True, tasks=None) == get_dual_tasks()
