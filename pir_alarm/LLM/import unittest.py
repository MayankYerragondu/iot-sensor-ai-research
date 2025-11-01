import unittest
from unittest.mock import patch, MagicMock
from pir_alarm.LLM.chatgpt import handle_user_query, parse_intent

class TestChatGPT(unittest.TestCase):

    @patch("pir_alarm.LLM.chatgpt.client")
    def test_parse_intent_returns_expected_json(self, mock_client):
        # Mock OpenAI response
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"intent": "rag", "device_id": "70:2c:1f:37:c3:b6", "time_range": "night", "query": "why trigger at night"}'
        mock_client.chat.completions.create.return_value = mock_resp

        result = parse_intent("Why does device 70:2c:1f:37:c3:b6 always trigger at night?")
        self.assertEqual(result["intent"], "rag")
        self.assertEqual(result["device_id"], "70:2c:1f:37:c3:b6")
        self.assertEqual(result["time_range"], "night")
        self.assertIn("trigger", result["query"])

    @patch("pir_alarm.LLM.chatgpt.parse_intent")
    @patch("pir_alarm.LLM.chatgpt.client")
    @patch("pir_alarm.LLM.chatgpt.rag_chain")
    def test_handle_user_query_returns_answer(self, mock_rag_chain, mock_client, mock_parse_intent):
        # Mock intent parsing
        mock_parse_intent.return_value = {
            "intent": "rag",
            "device_id": "70:2c:1f:37:c3:b6",
            "time_range": "night",
            "query": "why trigger at night"
        }
        # Mock RAG chain
        mock_rag_chain.invoke.return_value = "Device triggered frequently at night due to motion."
        # Mock OpenAI completion
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "The device triggers at night because of increased motion."
        mock_client.chat.completions.create.return_value = mock_resp

        answer = handle_user_query("Why does device 70:2c:1f:37:c3:b6 always trigger at night?")
        self.assertIn("triggers at night", answer)

if __name__ == "__main__":
    unittest.main()