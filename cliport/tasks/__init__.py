"""Ravens tasks."""

from cliport.tasks.align_box_corner import AlignBoxCorner
from cliport.tasks.align_rope import AlignRope
from cliport.tasks.assembling_kits import AssemblingKits
from cliport.tasks.assembling_kits_seq import AssemblingKitsSeq
from cliport.tasks.block_insertion import BlockInsertion
from cliport.tasks.extended_tasks import *
from cliport.tasks.generated_task import GeneratedTask
from cliport.tasks.manipulating_rope import ManipulatingRope
from cliport.tasks.move_blocks import MoveBlocksBetweenAbsolutePositions
from cliport.tasks.move_blocks import MoveBlocksBetweenAbsolutePositionsBySize
from cliport.tasks.packing_boxes import PackingBoxes
from cliport.tasks.packing_boxes_pairs import PackingBoxesPairs
from cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsSeq
from cliport.tasks.packing_shapes import PackingShapes
from cliport.tasks.palletizing_boxes import PalletizingBoxes
from cliport.tasks.place_red_in_green import PlaceRedInGreen
from cliport.tasks.put_block_in_bowl import PutBlockInMismatchingBowl
from cliport.tasks.put_block_in_bowl import PutBlockInMatchingBowl
from cliport.tasks.separating_piles import SeparatingPiles
from cliport.tasks.stack_block_pyramid import StackBlockPyramid
from cliport.tasks.stack_block_pyramid_seq import StackBlockPyramidSeq
from cliport.tasks.stack_blocks import StackAllBlocksOnAZone
from cliport.tasks.stack_blocks import StackBlocksOfSameSize
from cliport.tasks.stack_blocks import StackBlockOfSameColor
from cliport.tasks.stack_blocks import StackBlocksByColorAndSize
from cliport.tasks.stack_blocks import StackBlocksWithAlternateColor
from cliport.tasks.stack_blocks import StackBlocksByRelativePositionAndColor
from cliport.tasks.sweeping_piles import SweepingPiles
from cliport.tasks.task import Task
from cliport.tasks.towers_of_hanoi import TowersOfHanoi
from cliport.tasks.towers_of_hanoi_seq import TowersOfHanoiSeq

names = {
    # demo conditioned
    'align-box-corner': AlignBoxCorner,
    'assembling-kits': AssemblingKits,
    'assembling-kits-easy': AssemblingKitsEasy,
    'block-insertion': BlockInsertion,
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-translation': BlockInsertionTranslation,
    'manipulating-rope': ManipulatingRope,
    'packing-boxes': PackingBoxes,
    'palletizing-boxes': PalletizingBoxes,
    'place-red-in-green': PlaceRedInGreen,
    'stack-block-pyramid': StackBlockPyramid,
    'sweeping-piles': SweepingPiles,
    'towers-of-hanoi': TowersOfHanoi,
    'gen-task': GeneratedTask,

    # goal conditioned
    'align-rope': AlignRope,
    'assembling-kits-seq': AssemblingKitsSeq,
    'assembling-kits-seq-seen-colors': AssemblingKitsSeqSeenColors,
    'assembling-kits-seq-unseen-colors': AssemblingKitsSeqUnseenColors,
    'assembling-kits-seq-full': AssemblingKitsSeqFull,
    'packing-shapes': PackingShapes,
    'packing-boxes-pairs': PackingBoxesPairsSeenColors,
    'packing-boxes-pairs-seen-colors': PackingBoxesPairsSeenColors,
    'packing-boxes-pairs-unseen-colors': PackingBoxesPairsUnseenColors,
    'packing-boxes-pairs-full': PackingBoxesPairsFull,
    'packing-seen-google-objects-seq': PackingSeenGoogleObjectsSeq,
    'packing-unseen-google-objects-seq': PackingUnseenGoogleObjectsSeq,
    'packing-seen-google-objects-group': PackingSeenGoogleObjectsGroup,
    'packing-unseen-google-objects-group': PackingUnseenGoogleObjectsGroup,
    'put-block-in-bowl': PutBlockInBowlSeenColors,
    'put-block-in-bowl-seen-colors': PutBlockInBowlSeenColors,
    'put-block-in-bowl-unseen-colors': PutBlockInBowlUnseenColors,
    'put-block-in-bowl-full': PutBlockInBowlFull,
    'stack-block-pyramid-seq': StackBlockPyramidSeqSeenColors,
    'stack-block-pyramid-seq-seen-colors': StackBlockPyramidSeqSeenColors,
    'stack-block-pyramid-seq-unseen-colors': StackBlockPyramidSeqUnseenColors,
    'stack-block-pyramid-seq-full': StackBlockPyramidSeqFull,
    'separating-piles': SeparatingPilesSeenColors,
    'separating-piles-seen-colors': SeparatingPilesSeenColors,
    'separating-piles-unseen-colors': SeparatingPilesUnseenColors,
    'separating-piles-full': SeparatingPilesFull,
    'towers-of-hanoi-seq': TowersOfHanoiSeqSeenColors,
    'towers-of-hanoi-seq-seen-colors': TowersOfHanoiSeqSeenColors,
    'towers-of-hanoi-seq-unseen-colors': TowersOfHanoiSeqUnseenColors,
    'towers-of-hanoi-seq-full': TowersOfHanoiSeqFull,

    'stack-blocks-by-color-and-size': StackBlocksByColorAndSize,
    'stack-all-blocks-on-a-zone': StackAllBlocksOnAZone,
    'stack-blocks-of-same-size': StackBlocksOfSameSize,
    'stack-blocks-of-same-color': StackBlockOfSameColor,
    'stack-blocks-with-alternate-color': StackBlocksWithAlternateColor,
    'stack-blocks-by-relative-position-and-color': StackBlocksByRelativePositionAndColor,
    'move-blocks-between-absolute-positions': MoveBlocksBetweenAbsolutePositions,
    'move-blocks-between-absolute-positions-by-size': MoveBlocksBetweenAbsolutePositionsBySize,
    'put-block-into-mismatching-bowl': PutBlockInMismatchingBowl,
    'put-block-into-matching-bowl': PutBlockInMatchingBowl,
}

from cliport.generated_tasks import new_names

names.update(new_names)
from cliport.loho_tasks import new_names

names.update(new_names)
