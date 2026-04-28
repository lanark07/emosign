from app import config


class SentenceBuilder:
    """
    Accumulates ASL letters into a sentence with debounce logic.

    A letter is accepted only when the same prediction appears for
    ASL_DEBOUNCE_FRAMES consecutive frames and confidence >= ASL_CONF_THRESHOLD.
    Special tokens: 'del' removes the last character, 'space' appends a space,
    'nothing' is ignored.
    """

    def __init__(self):
        self._chars:        list[str] = []
        self._last_letter:  str       = ""
        self._hold_count:   int       = 0

    def add_prediction(self, letter: str, conf: float) -> bool:
        """
        Feed a raw prediction. Returns True if a new character was committed.
        """
        if letter == "nothing":
            self._reset_hold()
            return False

        if letter != self._last_letter:
            self._last_letter = letter
            self._hold_count  = 1
            return False

        self._hold_count += 1
        if self._hold_count < config.ASL_DEBOUNCE_FRAMES:
            return False

        # Hold threshold reached — commit and reset so the next distinct sign
        # requires a fresh hold rather than immediately appending again.
        self._hold_count = 0
        self._last_letter = ""

        if letter == "del":
            if self._chars:
                self._chars.pop()
            return True

        if letter == "space":
            self._chars.append(" ")
            return True

        if conf >= config.ASL_CONF_THRESHOLD:
            self._chars.append(letter)
            return True

        return False

    def get_sentence(self) -> str:
        return "".join(self._chars)

    def clear(self):
        self._chars.clear()
        self._reset_hold()

    def _reset_hold(self):
        self._last_letter = ""
        self._hold_count  = 0
