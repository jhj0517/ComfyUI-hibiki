from .nodes import *


#  Map all your custom nodes classes with the names that will be displayed in the UI.
NODE_CLASS_MAPPINGS = {
    "(Down)Load Hibiki Model": HibikiModelLoader,
    "Speech To Speech Translation": SpeechToSpeechTranslation,
    "Get Audio File Path": GetAudioFilePath,
}


__all__ = ['NODE_CLASS_MAPPINGS']
