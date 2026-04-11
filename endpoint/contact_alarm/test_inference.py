class DummyModel:
    def predict(self, X):
        return [42] * len(X)

    def score_samples(self, X):
        return [0.5] * len(X)

class DummyScaler:
    def transform(self, X):
        return X + 1

def create_joblib_files(tmpdir, with_scaler=True):
    model_path = os.path.join(tmpdir, "model.joblib")
    joblib.dump(DummyModel(), model_path)
    scaler_path = None
    if with_scaler:
        scaler_path = os.path.join(tmpdir, "model_scaler.joblib")
        joblib.dump(DummyScaler(), scaler_path)
    return model_path, scaler_path

def test_model_fn_loads_model_and_scaler(tmp_path):
    model_path, scaler_path = create_joblib_files(tmp_path, with_scaler=True)
    result = inference.model_fn(str(tmp_path))
    assert isinstance(result["model"], DummyModel)
    assert isinstance(result["scaler"], DummyScaler)

def test_model_fn_loads_model_without_scaler(tmp_path):
    model_path, _ = create_joblib_files(tmp_path, with_scaler=False)
    result = inference.model_fn(str(tmp_path))
    assert isinstance(result["model"], DummyModel)
    assert result["scaler"] is None

def test_model_fn_raises_if_no_model(tmp_path):
    # No joblib files at all
    with pytest.raises(FileNotFoundError):
        inference.model_fn(str(tmp_path))

def test_input_fn_parses_json_with_previous_timestamp():
    ts = "2024-06-01T12:34:56Z"
    prev_ts = "2024-06-01T12:00:00Z"
    body = json.dumps({
        "timestamp": ts,
        "previous_timestamp": prev_ts
    })