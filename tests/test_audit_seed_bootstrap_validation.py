import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1]/'docs/audit_seed_bootstrap_validation.py'


def auditor():
    assert MODULE.is_file(), 'seed bootstrap auditor not implemented'
    spec=importlib.util.spec_from_file_location('seed_bootstrap_audit_test',MODULE)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_audit_accepts_skipped_holdout_but_rejects_missing_evidence(tmp_path):
    m=auditor()
    record={'status':'not_applicable','reason':'diagnostic_gate_not_passed','generation_nfe':0,
            'cases':[{'case_id':'one','status':'not_applicable','reason':'diagnostic_gate_not_passed'}]}
    assert m.validate_holdout_record(record,{'eligible':False},['one'])=='not_applicable'
    with pytest.raises(ValueError,match='eligibility'):
        m.validate_holdout_record(record,{'eligible':True},['one'])
    with pytest.raises(ValueError,match='case'):
        m.validate_holdout_record(record,{'eligible':False},['one','two'])
    plan=tmp_path/'experiment.json'
    plan.write_text('{"seed_bootstrap_schema":1,"source_files":{},"output_root":"'+tmp_path.as_posix()+'"}')
    with pytest.raises((FileNotFoundError,ValueError)):
        m.audit(plan)


def test_audit_rejects_incomplete_or_changed_run(tmp_path):
    m=auditor();item=tmp_path/'evidence';item.write_bytes(b'original')
    row={'status':'complete','file_sha256':{str(item):m.sha(item)},'numeric_checks':{'routing':True},'generation_nfe':16}
    assert m.validate_run_files(row)==16
    row['numeric_checks']['routing']=False
    with pytest.raises(ValueError,match='numeric'):m.validate_run_files(row)
    row['numeric_checks']['routing']=True;item.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed'):m.validate_run_files(row)


def test_holdout_failure_keeps_unknown_inference_cost():
    m=auditor()
    record={'status':'executed','cases':[{'case_id':'one','status':'failed','reason':'history error','generation_nfe':None}],
        'unknown_failure_nfe':True}
    assert m.validate_holdout_record(record,{'eligible':True},['one'])=='executed_with_failure'
