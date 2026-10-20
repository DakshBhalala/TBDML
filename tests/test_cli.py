"""Unit tests for the CompactTab Typer CLI commands."""

from typer.testing import CliRunner

from compacttab.cli import app

runner = CliRunner()


def test_cli_env_info():
    result = runner.invoke(app, ["env-info"])
    assert result.exit_code == 0
    assert "Python Version" in result.stdout
    assert "PyTorch Version" in result.stdout


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "smoke-benchmark" in result.stdout
    assert "env-info" in result.stdout
