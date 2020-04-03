import numpy as np
from .computing_utils import largest_indices, smallest_indices


class Expert:
    """This class sorts the expert units determined by a score computed
        by a given method (Influence, Activation, etc.)

    Arguments:
        unit {string} -- The type of the expert unit: nueron, channel or
            location.

    Raises:
        TypeError: if unit is not a string, raise this error.
        ValueError: if unit is not one of 'Nueron', 'Channel' or 'Location',
            raise this error.
    """

    def __init__(self, unit):
        if not isinstance(unit, str):
            raise TypeError("unit must be a string")
        unit = unit.lower()
        if unit not in ['neuron', 'channel', 'location']:
            raise ValueError(
                "unit can only be a string of 'neuron', 'channel' or 'location'"
            )
        self.unit = unit
        self._tops = None
        self._bottoms = None
        self._eg = None
        self._ag = None
        self.ave_scores = None

    def get_tops(self):
        """Returns: np.Array -- A numpy.Array of experts """
        return self._tops

    def get_bottoms(self):
        """Returns: np.Array -- A numpy.Array of non-experts """
        return self._bottoms

    def get_expert_group(self):
        """Returns: np.Array -- A numpy.Array of experts with positive influence """
        return self._eg

    def get_antiexpert_group(self):
        """Returns: np.Array -- A numpy.Array of experts with negative influence """
        return self._ag

    def get_unit(self):
        """Returns: str -- The type of experts """
        return self.unit

    def get_avescores(self):
        """Returns: np.ndarray --- The average scores over the dataset """
        return self.ave_scores

    def get_inter_unit_wts(self, top_idx):
        """get_inter_unit_wts Compute the inter unit weights given its indices.

        Arguments:
            top_idx {np.ndarray}} -- The indices of experts

        Raises:
            ValueError: Find scores for neuron experts but provided idx can not locate a neuron.
            ValueError: Find scores for location experts but provided idx can not locate a location.
            ValueError: Find scores for neuron/channel experts but provided idx can not locate a neuron/channel.
            ValueError: Could not find average scores

        Returns:
            np.ndarray -- Weights of chosen experts
        """
        if self.ave_scores is not None:
            if len(self.ave_scores.shape) == 3:
                multi_unit_score = self.ave_scores[self._tops[top_idx].T[0],
                                                   self._tops[top_idx].T[1],
                                                   self._tops[top_idx].T[2]]
            if len(self.ave_scores.shape) == 2:
                multi_unit_score = self.ave_scores[self._tops[top_idx].T[0],
                                                   self._tops[top_idx].T[1]]
            if len(self.ave_scores.shape) == 1:
                multi_unit_score = self.ave_scores[self._tops[top_idx]]

            wts = multi_unit_score / (np.sum(multi_unit_score) + 1e-9)
            return wts
        else:
            raise ValueError("Could not find average scores")

    def __call__(self, raw_scores, heuristic='max', channel_first=True):
        """Find the expert or the non-experts.

        Arguments:
            raw_scores {np.ndarray} -- A np.Array containing the scores (e.g. Influence)
                The shape pf the array should be either NxK (for FC layer)
                or NxCxHxW (for CONV layer)

        Keyword Arguments:
            heuristic {str} -- The heuristic of sorting applied on the raw scores.
                Currently support: max, mean, min                                                   (default: {'max'})
            channel_first {bool} -- Indicate whether the raw scores have a channel-first order.
                Ignore this if the raw scores come from a FC layer                                  (default: {True})

        Raises:
            ValueError: If the dimension of the raw scores is not 2 or 4, raise this error.
            ValueError: If a heuristic is not max, mean or min, raise this error.
            ValueError: If unit is Channel but raw scores come from a FC layer, raise this error.
            NotImplementedError: If raw scores are not np.Array, raise this error.
        """
        if isinstance(raw_scores, np.ndarray):
            if len(raw_scores.shape) == 2:
                layer_type = 'fc'
            elif len(raw_scores.shape) == 4:
                layer_type = 'conv'
            else:
                raise ValueError(
                    "The dimension of raw scores should be 2 or 4, but got %d"
                    % len(raw_scores))

            if self.unit == 'neuron':
                ave_scores = np.mean(raw_scores, axis=0)
                self.ave_scores = ave_scores
                if layer_type == 'fc':
                    self._bottoms = np.argsort(ave_scores)
                    self._tops = self._bottoms[::-1]
                elif layer_type == 'conv':
                    self._bottoms = smallest_indices(
                        ave_scores, ave_scores.shape[0] * ave_scores.shape[1] *
                        ave_scores.shape[2] - 1).T
                    self._tops = largest_indices(
                        ave_scores, ave_scores.shape[0] * ave_scores.shape[1] *
                        ave_scores.shape[2] - 1).T

            elif self.unit == 'channel':
                if layer_type == 'conv':
                    if not channel_first:
                        raw_scores = np.transpose(raw_scores, (0, 3, 1, 2))
                    if heuristic == 'max':
                        scores = np.max(raw_scores, axis=(2, 3))
                    elif heuristic == 'min':
                        scores = np.min(raw_scores, axis=(2, 3))
                    elif heuristic == 'mean':
                        scores = np.mean(raw_scores, axis=(2, 3))
                    else:
                        raise ValueError(
                            "heuristic can only be max, min or mean.")

                    ave_scores = np.mean(scores, axis=0)
                    self.ave_scores = ave_scores
                    self._bottoms = np.argsort(ave_scores)
                    self._tops = self._bottoms[::-1]

                else:
                    raise ValueError(
                        "Channel Influence does not support fully connected layer"
                    )
        else:
            raise NotImplementedError(
                "Only numpy.Array is supported currently")

    def expert_group(self, threshold=0):
        """expert_group Seperate the experts into expert group and anti-expert group


        Keyword Arguments:
            threshold {int} -- The threshold to seperate (default: {0})

        Raises:
            ValueError: The average influence scores are not computed yet

        Returns:
            tuple -- A tuple of np.adarray. The indices of units of expert group
            and units in anti-expert group
        """
        if self.ave_scores is None:
            raise ValueError("Please compute the average scores first \
            or provide one using the keyword 'ave_scores'")

        pointer = len(self._bottoms) // 2
        prev_score = None
        neg_idx = None
        pos_idx = None

        while pointer >= 0 and pointer < len(self._bottoms):
            unit_id = self._bottoms[pointer]

            if len(self._bottoms.shape) == 1:
                score = self.ave_scores[unit_id]
            else:
                score = self.ave_scores[unit_id[0], unit_id[1], unit_id[2]]

            if prev_score is not None:
                if prev_score < threshold and score >= threshold:
                    neg_idx = self._bottoms[:pointer]
                    pos_idx = self._bottoms[pointer:]
                    break
                elif prev_score > threshold and score <= threshold:
                    neg_idx = self._bottoms[:pointer + 1]
                    pos_idx = self._bottoms[pointer + 1:]
                    break
                elif prev_score < threshold and score < threshold:
                    prev_score = score
                    pointer += 1
                elif prev_score > threshold and score > threshold:
                    prev_score = score
                    pointer -= 1
            else:
                prev_score = score
                pointer += 1

        return pos_idx[::-1], neg_idx
