"""Emotion detection module using Watson NLP service."""

import requests


URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)

HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def empty_response():
    """Return an empty emotion response."""
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None
    }


def emotion_detector(text_to_analyze):
    """Analyze the emotion of the provided text.

    Args:
        text_to_analyze (str): Text that will be analyzed.

    Returns:
        dict: Emotion scores and the dominant emotion.
    """
    if not text_to_analyze or not text_to_analyze.strip():
        return empty_response()

    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    try:
        response = requests.post(
            URL,
            json=payload,
            headers=HEADERS,
            timeout=30
        )
    except requests.RequestException:
        return empty_response()

    if response.status_code == 400:
        return empty_response()

    response_json = response.json()
    emotions = response_json["emotionPredictions"][0]["emotion"]

    emotion_scores = {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"]
    }

    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    emotion_scores["dominant_emotion"] = dominant_emotion

    return emotion_scores