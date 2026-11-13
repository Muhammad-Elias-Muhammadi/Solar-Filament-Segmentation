"""Instance segmentation of solar filaments in GONG H-alpha frames.

The pipeline in one line: a U-Net predicts, for every pixel of the 2048 x 2048 frame, the
probability that an annotator marks it; connected components of that map are the candidate
filaments; two scorers estimate how well each candidate will match the annotators, and only the
candidates worth keeping for Panoptic Quality are written to the submission.
"""
