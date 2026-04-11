import unittest
from unittest.mock import patch, MagicMock

class TestEndpointDeploy(unittest.TestCase):
    @patch("boto3.client")
    @patch("sagemaker.Session")
    @patch("sagemaker.model.Model")
    @patch("sagemaker.multidatamodel.MultiDataModel")
    def test_endpoint_deploy_flow(
        self, mock_MultiDataModel, mock_Model, mock_Session, mock_boto3_client
    ):
        # Mock boto3 client and its exceptions
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.exceptions.ClientError = Exception

        # Mock Model and MultiDataModel
        mock_model_instance = MagicMock()
        mock_Model.return_value = mock_model_instance

        mock_mme_instance = MagicMock()
        mock_MultiDataModel.return_value = mock_mme_instance
        mock_predictor = MagicMock()
        mock_mme_instance.deploy.return_value = mock_predictor

        # Import the code under test
        import endpoint.env_sensor.endpoint_deploy

        # Check cleanup calls
        self.assertTrue(mock_client.delete_endpoint.called)
        self.assertTrue(mock_client.delete_endpoint_config.called)
        self.assertTrue(mock_client.delete_model.called)

        # Check Model instantiation
        mock_Model.assert_called_once()
        args, kwargs = mock_Model.call_args
        self.assertIn("image_uri", kwargs)
        self.assertIn("role", kwargs)
        self.assertIn("entry_point", kwargs)
        self.assertIn("env", kwargs)
        self.assertIn("sagemaker_session", kwargs)

        # Check MultiDataModel instantiation
        mock_MultiDataModel.assert_called_once()
        args, kwargs = mock_MultiDataModel.call_args
        self.assertIn("name", kwargs)
        self.assertIn("model_data_prefix", kwargs)
        self.assertIn("model", kwargs)
        self.assertIn("sagemaker_session", kwargs)

        # Check deploy call
        mock_mme_instance.deploy.assert_called_once_with(
            initial_instance_count=1,
            instance_type="ml.m5.xlarge",
            endpoint_name="env-sensor-mme-endpoint-2",
            update_endpoint=True,
        )

if __name__ == "__main__":
    unittest.main()