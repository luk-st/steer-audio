"""
Prompt construction utilities for computing steering vectors in ACE-Step.

Adapted from CASteer for audio/music generation.
"""


def get_music_descriptions(num=50):
    """
    Get diverse music descriptions for steering vector computation.
    Similar to ImageNet classes but for music.
    """
    music_descriptions = [
        "a song",
        "a melody",
        "music",
        "a tune",
        "a track",
        "a composition",
        "instrumental music",
        "a piece of music",
        "background music",
        "a musical performance",
        "an upbeat song",
        "a slow song",
        "a fast-paced track",
        "electronic music",
        "acoustic music",
        "orchestral music",
        "a pop song",
        "a rock song",
        "a jazz piece",
        "a classical piece",
        "hip hop music",
        "country music",
        "blues music",
        "folk music",
        "reggae music",
        "metal music",
        "punk rock",
        "dance music",
        "ambient music",
        "lofi music",
        "a ballad",
        "a love song",
        "a happy song",
        "a sad song",
        "energetic music",
        "calm music",
        "dramatic music",
        "cheerful music",
        "melancholic music",
        "aggressive music",
        "gentle music",
        "powerful music",
        "soft music",
        "loud music",
        "rhythmic music",
        "harmonious music",
        "dissonant music",
        "simple music",
        "complex music",
        "minimalist music",
    ]
    return music_descriptions[:num]


def get_prompts_instrument(num=50, concept_pos='piano', concept_neg=None):
    """
    Generate prompt pairs for instrument-based steering.

    Args:
        num: Number of prompt pairs to generate
        concept_pos: Target instrument (e.g., 'piano', 'guitar', 'drums')
        concept_neg: Source instrument (if None, uses neutral prompts)

    Returns:
        prompts_pos: List of prompts with target instrument
        prompts_neg: List of prompts with source instrument or neutral
    """
    music_descriptions = get_music_descriptions(num)

    prompts_pos = []
    prompts_neg = []
    for desc in music_descriptions:
        prompts_pos.append(f"{desc} with {concept_pos}")
        if concept_neg is not None:
            prompts_neg.append(f"{desc} with {concept_neg}")
        else:
            prompts_neg.append(desc)

    return prompts_pos, prompts_neg


def get_prompts_electronic_music(num=50, concept_pos='rock_genre', concept_neg=None):
    """
    Generate prompt pairs for genre-based steering.

    Args:
        num: Number of prompt pairs to generate
        concept_pos: Target genre (e.g., 'rock_genre', 'rock', 'classical')
        concept_neg: Source genre (if None, uses neutral prompts)

    Returns:
        prompts_pos: List of prompts with target genre
        prompts_neg: List of prompts with source genre or neutral
    """
    music_descriptions = get_music_descriptions(num)

    prompts_pos = []
    prompts_neg = []
    for desc in music_descriptions:
        prompts_pos.append(f"{desc}, {concept_pos} style")
        if concept_neg is not None:
            prompts_neg.append(f"{desc}, {concept_neg} style")
        else:
            prompts_neg.append(desc)

    return prompts_pos, prompts_neg


def get_prompts_gender(concept_pos='female vocals', concept_neg='male vocals'):
    """
    Generate prompt pairs for vocal gender steering.

    Args:
        concept_pos: Target vocal type (e.g., 'female vocals', 'female singer')
        concept_neg: Source vocal type (e.g., 'male vocals', 'male singer')

    Returns:
        prompts_pos: List of prompts with target vocals
        prompts_neg: List of prompts with source vocals
    """
    genres = ['pop', 'rock', 'rock_genre', 'blues', 'country', 'r&b', 'soul', 'folk']
    tempos = ['upbeat', 'slow', 'mid-tempo', 'ballad', 'energetic', 'mellow']
    moods = ['happy', 'sad', 'melancholic', 'uplifting', 'romantic', 'powerful']

    prompts_pos = []
    prompts_neg = []

    for genre in genres:
        for tempo in tempos:
            prompts_pos.append(f"a {tempo} {genre} song with {concept_pos}")
            prompts_neg.append(f"a {tempo} {genre} song with {concept_neg}")

    for genre in genres:
        for mood in moods:
            prompts_pos.append(f"a {mood} {genre} song with {concept_pos}")
            prompts_neg.append(f"a {mood} {genre} song with {concept_neg}")

    return prompts_pos, prompts_neg


def get_prompts_tempo(concept_pos='fast tempo', concept_neg='slow tempo'):
    """
    Generate prompt pairs for tempo-based steering.

    Args:
        concept_pos: Target tempo (e.g., 'fast tempo', 'upbeat')
        concept_neg: Source tempo (e.g., 'slow tempo', 'ballad')

    Returns:
        prompts_pos: List of prompts with target tempo
        prompts_neg: List of prompts with source tempo
    """
    genres = ['pop', 'rock', 'rock_genre', 'electronic', 'classical', 'hip hop', 'country', 'blues']
    instruments = ['piano', 'guitar', 'drums', 'synthesizer', 'violin', 'saxophone']

    prompts_pos = []
    prompts_neg = []

    for genre in genres:
        prompts_pos.append(f"{genre} music, {concept_pos}")
        prompts_neg.append(f"{genre} music, {concept_neg}")

    for instrument in instruments:
        prompts_pos.append(f"music with {instrument}, {concept_pos}")
        prompts_neg.append(f"music with {instrument}, {concept_neg}")

    return prompts_pos, prompts_neg


def get_prompts_mood(concept_pos='happy', concept_neg='sad'):
    """
    Generate prompt pairs for mood-based steering.

    Args:
        concept_pos: Target mood (e.g., 'happy', 'energetic')
        concept_neg: Source mood (e.g., 'sad', 'calm')

    Returns:
        prompts_pos: List of prompts with target mood
        prompts_neg: List of prompts with source mood
    """
    music_descriptions = get_music_descriptions(30)

    prompts_pos = []
    prompts_neg = []

    for desc in music_descriptions:
        prompts_pos.append(f"{concept_pos} {desc}")
        prompts_neg.append(f"{concept_neg} {desc}")

    return prompts_pos, prompts_neg
