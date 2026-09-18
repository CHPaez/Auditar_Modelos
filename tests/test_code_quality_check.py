from code_quality_check import run


def test_run_captures_command_output_and_exit_code():
    result = run(["python", "--version"])

    assert result["exit_code"] == 0
    assert result["output"] != ""
