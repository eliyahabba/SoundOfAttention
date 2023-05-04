class Interface:
    """
    A class representing an interface for exploring the correlation between the attention layers of an Audio and Text model
    """

    def __init__(self, audio_model, text_model):
        """
        Initialize the interface with the provided audio and text models

        Parameters:
        audio_model (AudioModel): The audio model
        text_model (LanguageModel): The text model
        """
        self.audio_model = audio_model
        self.text_model = text_model

    def display_attention_matrices(self, audio, text):
        """
        Display the attention matrices for the provided audio and text samples

        Parameters:
        audio (np.ndarray): A numpy array representing the audio sample
        text (str): The input text
        """
        pass