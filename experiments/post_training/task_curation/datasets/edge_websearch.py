# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert Edge's SFT preparation plans with a separately declared planning judge."""

import json
from dataclasses import replace

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.grader import verifyit_package
from taskcompendium.models import TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizationChange, NormalizedTask, RawRow
from taskcompendium.pipeline.reviewer_spec import JudgeTier
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import RUBRIC_CHECKLIST

from experiments.post_training.task_curation.judges import JUDGE_TIERS
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

REPO = "yatin-superintelligence/Edge-Agent-Reasoning-WebSearch-260K"
REVISION = "7e8e455fff52e6d21dce4ce4a5a1bddd13031e1a"
SOURCE_ID = "Yatin:Edge-Agent-Reasoning-WebSearch-260K"
VERSION = "edge-websearch-preparation-judge-v1"
NATIVE_FIELDS = frozenset({"batch_index_id", "role", "industry", "os", "user_prompt", "agent_reasoning"})
CARD_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md"
PREPARATION_CONTRACT = (
    "You are a preparatory router for a downstream executor. Produce a preparation plan for the simulated "
    "user request below, rather than carrying out its requested actions. Do not claim to have searched, "
    "executed commands or verified facts. Cover the objective and constraints; known facts versus assumptions; "
    "material ambiguities and useful clarification questions; a concrete verification checklist; and proposed "
    "targeted web-search queries. State when an aspect has no material relevance instead of inventing a gap. "
    "Use the supplied source context. No particular section heading, word count or number of queries is required."
)
CRITERIA = (
    "The preparation plan correctly identifies the simulated request's objective and explicit constraints, "
    "respecting the supplied role, industry and operating-system context without pretending to execute the task.",
    "The plan distinguishes grounded facts from assumptions and identifies material knowledge gaps without "
    "unsupported certainty or fabricated results.",
    "The plan identifies material ambiguities and useful clarification questions where they exist; it does not "
    "invent requirements or penalize an already clear request merely for lacking ambiguities.",
    "The verification checklist names concrete facts, dependencies or authoritative sources needed to prepare "
    "execution, tailored to the actual request rather than generic boilerplate.",
    "The proposed search queries target the request's actual verification needs and are useful to downstream "
    "retrieval. They are plans, not claims that a live search or tool call occurred.",
)


def convert_plan(row: RawRow, _context: ConversionContext) -> NormalizedTask | ImportRejection:
    native = dict(row.data)
    if (
        set(native) != NATIVE_FIELDS
        or type(native["batch_index_id"]) is not int
        or any(
            not isinstance(native[field], str) or not native[field].strip()
            for field in NATIVE_FIELDS - {"batch_index_id"}
        )
    ):
        return source_defect("incomplete_edge_native_contract", "Expected the six original fields with nonempty text")
    metadata = {field: native[field] for field in ("role", "industry", "os")}
    instruction = PREPARATION_CONTRACT + "\n\nSource context:\n" + json.dumps(metadata, ensure_ascii=False)
    events = (
        TextMessage(role="system", content=instruction),
        TextMessage(role="user", content=native["user_prompt"]),
    )
    spec = replace(
        JUDGE_TIERS[JudgeTier.MEDIUM].spec,
        rubric=RUBRIC_CHECKLIST,
        criteria=CRITERIA,
        context="conversation.json",
        question="Evaluate the candidate preparation plan against the source-derived planning contract and request.",
        references=(),
        system_prompt="",
        prompt_template="",
        label_scores={},
        exact_gate=False,
    )
    package = verifyit_package(
        spec,
        (
            inline_resource(
                "conversation.json", json.dumps([event.model_dump(mode="json") for event in events]).encode()
            ),
            inline_resource("source-row.json", json.dumps(native, ensure_ascii=False).encode()),
        ),
    )
    task = conversation_task(row, events=events, package=package).model_copy(
        update={
            "tags": ("train", "preparation-planning", "prompt-only-judge"),
            "labels": {"source_id": SOURCE_ID, "native_batch_index": str(native["batch_index_id"])},
        }
    )
    change = NormalizationChange(
        field="context.events[0]",
        reason=f"Declare the preparatory-router task described in the pinned source card: {CARD_URL}",
        original="No system instruction is stored in the native six-field row; the task scope is described by its card.",
        replacement=instruction,
    )
    return NormalizedTask(task, (change,))


def source(files: tuple[str, ...]) -> RlDataSource[CurationRecipe]:
    return RlDataSource(
        info=SourceInfo(
            origin=SourceOrigin.YATIN,
            upstream_dataset="Edge-Agent-Reasoning-WebSearch-260K",
            upstream_name="Edge-Agent-Reasoning-WebSearch-260K",
            build_path=VERSION,
            family="preparation-planning",
            data_restriction=DataRestriction.OPEN,
            task_format=TaskFormat.TASKSPEC,
            count=260293,
            notes=(
                "New MEDIUM checklist judge for the preparation-planning task described by the pinned source card. "
                "An explicit source-derived system contract is recorded as normalization; raw user_prompt and "
                "role/industry/OS strings are unchanged. Generated agent_reasoning stays private provenance, "
                "never judge gold or native oracle. There is no native executable grader or live search tool. "
                "All 267 data shards contain 260,293 rows; four non-data files are separately accounted for. "
                "The card declares 263,098 rows, a 2,805-row metadata discrepancy. Generator checkpoint identity "
                "is undeclared. No conversion or grading quality claim follows from those source descriptions."
            ),
        ),
        config=CurationRecipe(
            name="edge-websearch-preparation",
            version=VERSION,
            source=HfSource(REPO, REVISION, files, SourceFormat.PARQUET),
            convert=convert_plan,
            intended_use=IntendedUse.TRAIN,
        ),
        pipeline=process_rows,
    )

# The pinned revision's 267 data shards, enumerated from
# yatin-superintelligence/Edge-Agent-Reasoning-WebSearch-260K@7e8e455fff52e6d21dce4ce4a5a1bddd13031e1a;
# the declaration reads every shard at REVISION.
EDGE_DATA_FILES = (
    'edge_reasoning_train_0_296.parquet', 'edge_reasoning_train_100157_102262.parquet', 'edge_reasoning_train_10017_30995.parquet', 'edge_reasoning_train_101275_103537.parquet',
    'edge_reasoning_train_10184_11673.parquet', 'edge_reasoning_train_102680_104514.parquet', 'edge_reasoning_train_103097_113845.parquet', 'edge_reasoning_train_103816_105471.parquet',
    'edge_reasoning_train_105033_106484.parquet', 'edge_reasoning_train_106016_107518.parquet', 'edge_reasoning_train_106942_109362.parquet', 'edge_reasoning_train_108412_110689.parquet',
    'edge_reasoning_train_109140_139749.parquet', 'edge_reasoning_train_109785_111730.parquet', 'edge_reasoning_train_11019_12791.parquet', 'edge_reasoning_train_111925_114325.parquet',
    'edge_reasoning_train_113394_115588.parquet', 'edge_reasoning_train_114152_140999.parquet', 'edge_reasoning_train_114778_116554.parquet', 'edge_reasoning_train_116030_117561.parquet',
    'edge_reasoning_train_117077_118551.parquet', 'edge_reasoning_train_117746_119556.parquet', 'edge_reasoning_train_119229_120591.parquet', 'edge_reasoning_train_119720_121623.parquet',
    'edge_reasoning_train_120940_122676.parquet', 'edge_reasoning_train_121969_123664.parquet', 'edge_reasoning_train_12282_13862.parquet', 'edge_reasoning_train_122832_124668.parquet',
    'edge_reasoning_train_124162_125655.parquet', 'edge_reasoning_train_125113_126661.parquet', 'edge_reasoning_train_126203_127688.parquet', 'edge_reasoning_train_126930_128704.parquet',
    'edge_reasoning_train_128056_129718.parquet', 'edge_reasoning_train_129029_130712.parquet', 'edge_reasoning_train_130158_131695.parquet', 'edge_reasoning_train_131266_132724.parquet',
    'edge_reasoning_train_132255_133702.parquet', 'edge_reasoning_train_13281_14951.parquet', 'edge_reasoning_train_133194_134695.parquet', 'edge_reasoning_train_134230_135765.parquet',
    'edge_reasoning_train_135273_136762.parquet', 'edge_reasoning_train_136111_137755.parquet', 'edge_reasoning_train_137309_138753.parquet', 'edge_reasoning_train_1376_5163.parquet',
    'edge_reasoning_train_138276_141625.parquet', 'edge_reasoning_train_140016_180453.parquet', 'edge_reasoning_train_140742_142777.parquet', 'edge_reasoning_train_141241_189820.parquet',
    'edge_reasoning_train_141829_144019.parquet', 'edge_reasoning_train_143060_145055.parquet', 'edge_reasoning_train_14407_16051.parquet', 'edge_reasoning_train_144540_146049.parquet',
    'edge_reasoning_train_145194_147047.parquet', 'edge_reasoning_train_146328_148051.parquet', 'edge_reasoning_train_147549_149048.parquet', 'edge_reasoning_train_148505_150055.parquet',
    'edge_reasoning_train_149128_151043.parquet', 'edge_reasoning_train_150534_152061.parquet', 'edge_reasoning_train_151170_153054.parquet', 'edge_reasoning_train_152577_154064.parquet',
    'edge_reasoning_train_153630_155056.parquet', 'edge_reasoning_train_154250_156078.parquet', 'edge_reasoning_train_15532_17140.parquet', 'edge_reasoning_train_155704_157067.parquet',
    'edge_reasoning_train_156515_158105.parquet', 'edge_reasoning_train_157489_159104.parquet', 'edge_reasoning_train_158581_160093.parquet', 'edge_reasoning_train_159602_161070.parquet',
    'edge_reasoning_train_160660_162067.parquet', 'edge_reasoning_train_161737_163085.parquet', 'edge_reasoning_train_162412_164131.parquet', 'edge_reasoning_train_163521_165087.parquet',
    'edge_reasoning_train_164157_166131.parquet', 'edge_reasoning_train_165163_167156.parquet', 'edge_reasoning_train_1656_4020.parquet', 'edge_reasoning_train_166584_168162.parquet',
    'edge_reasoning_train_167543_169148.parquet', 'edge_reasoning_train_168810_170144.parquet', 'edge_reasoning_train_16942_18247.parquet', 'edge_reasoning_train_169689_171179.parquet',
    'edge_reasoning_train_170660_172160.parquet', 'edge_reasoning_train_171416_173166.parquet', 'edge_reasoning_train_172841_174205.parquet', 'edge_reasoning_train_173253_175213.parquet',
    'edge_reasoning_train_174133_176194.parquet', 'edge_reasoning_train_175384_177248.parquet', 'edge_reasoning_train_176334_178255.parquet', 'edge_reasoning_train_17683_19335.parquet',
    'edge_reasoning_train_177506_179236.parquet', 'edge_reasoning_train_178642_180290.parquet', 'edge_reasoning_train_179458_181597.parquet', 'edge_reasoning_train_180706_182884.parquet',
    'edge_reasoning_train_180897_190777.parquet', 'edge_reasoning_train_181924_183949.parquet', 'edge_reasoning_train_182974_184962.parquet', 'edge_reasoning_train_184396_185957.parquet',
    'edge_reasoning_train_185420_186977.parquet', 'edge_reasoning_train_186372_187966.parquet', 'edge_reasoning_train_186998_188993.parquet', 'edge_reasoning_train_18750_20506.parquet',
    'edge_reasoning_train_188466_189638.parquet', 'edge_reasoning_train_190308_191787.parquet', 'edge_reasoning_train_191318_192835.parquet', 'edge_reasoning_train_191869_193807.parquet',
    'edge_reasoning_train_193050_194801.parquet', 'edge_reasoning_train_194340_195777.parquet', 'edge_reasoning_train_195252_196799.parquet', 'edge_reasoning_train_196123_197785.parquet',
    'edge_reasoning_train_197269_198811.parquet', 'edge_reasoning_train_198258_199810.parquet', 'edge_reasoning_train_199215_200801.parquet', 'edge_reasoning_train_19940_21605.parquet',
    'edge_reasoning_train_200273_201801.parquet', 'edge_reasoning_train_201441_202817.parquet', 'edge_reasoning_train_202332_203813.parquet', 'edge_reasoning_train_203245_204814.parquet',
    'edge_reasoning_train_204297_205787.parquet', 'edge_reasoning_train_205402_206831.parquet', 'edge_reasoning_train_206256_207846.parquet', 'edge_reasoning_train_206829_208815.parquet',
    'edge_reasoning_train_208261_209820.parquet', 'edge_reasoning_train_209371_210827.parquet', 'edge_reasoning_train_210321_211851.parquet', 'edge_reasoning_train_21047_22809.parquet',
    'edge_reasoning_train_211284_212822.parquet', 'edge_reasoning_train_212364_213852.parquet', 'edge_reasoning_train_214503_215862.parquet', 'edge_reasoning_train_215400_216830.parquet',
    'edge_reasoning_train_216299_217847.parquet', 'edge_reasoning_train_21724_23889.parquet', 'edge_reasoning_train_217491_219039.parquet', 'edge_reasoning_train_218029_220042.parquet',
    'edge_reasoning_train_219435_221053.parquet', 'edge_reasoning_train_220329_222048.parquet', 'edge_reasoning_train_221150_223056.parquet', 'edge_reasoning_train_222464_224056.parquet',
    'edge_reasoning_train_223441_225027.parquet', 'edge_reasoning_train_22393_112758.parquet', 'edge_reasoning_train_224218_226081.parquet', 'edge_reasoning_train_225038_227079.parquet',
    'edge_reasoning_train_226213_228086.parquet', 'edge_reasoning_train_226943_229106.parquet', 'edge_reasoning_train_22770_24974.parquet', 'edge_reasoning_train_228161_230073.parquet',
    'edge_reasoning_train_229638_231102.parquet', 'edge_reasoning_train_230087_232142.parquet', 'edge_reasoning_train_231033_233121.parquet', 'edge_reasoning_train_232347_234109.parquet',
    'edge_reasoning_train_233605_235123.parquet', 'edge_reasoning_train_234577_236153.parquet', 'edge_reasoning_train_235456_237111.parquet', 'edge_reasoning_train_236174_238135.parquet',
    'edge_reasoning_train_237322_239137.parquet', 'edge_reasoning_train_238630_240121.parquet', 'edge_reasoning_train_239670_241134.parquet', 'edge_reasoning_train_240628_242121.parquet',
    'edge_reasoning_train_241614_243168.parquet', 'edge_reasoning_train_242634_244159.parquet', 'edge_reasoning_train_243123_245146.parquet', 'edge_reasoning_train_24368_26084.parquet',
    'edge_reasoning_train_24448_37814.parquet', 'edge_reasoning_train_244774_246142.parquet', 'edge_reasoning_train_245792_247150.parquet', 'edge_reasoning_train_246323_248140.parquet',
    'edge_reasoning_train_247693_249167.parquet', 'edge_reasoning_train_248646_250157.parquet', 'edge_reasoning_train_249630_251145.parquet', 'edge_reasoning_train_250629_252131.parquet',
    'edge_reasoning_train_251684_253144.parquet', 'edge_reasoning_train_252604_254155.parquet', 'edge_reasoning_train_253281_255175.parquet', 'edge_reasoning_train_25402_27184.parquet',
    'edge_reasoning_train_254308_256163.parquet', 'edge_reasoning_train_255646_257151.parquet', 'edge_reasoning_train_256845_258146.parquet', 'edge_reasoning_train_257800_259149.parquet',
    'edge_reasoning_train_258327_260141.parquet', 'edge_reasoning_train_259722_261159.parquet', 'edge_reasoning_train_260601_262165.parquet', 'edge_reasoning_train_261704_263161.parquet',
    'edge_reasoning_train_262649_264171.parquet', 'edge_reasoning_train_263654_264283.parquet', 'edge_reasoning_train_2642_80413.parquet', 'edge_reasoning_train_26555_28316.parquet',
    'edge_reasoning_train_27713_29418.parquet', 'edge_reasoning_train_28693_30523.parquet', 'edge_reasoning_train_29753_31646.parquet', 'edge_reasoning_train_2_2457.parquet',
    'edge_reasoning_train_30973_32771.parquet', 'edge_reasoning_train_31648_33858.parquet', 'edge_reasoning_train_33288_35035.parquet', 'edge_reasoning_train_342_9675.parquet',
    'edge_reasoning_train_34339_36275.parquet', 'edge_reasoning_train_3481_6175.parquet', 'edge_reasoning_train_35222_37448.parquet', 'edge_reasoning_train_35281_42291.parquet',
    'edge_reasoning_train_36623_39059.parquet', 'edge_reasoning_train_38407_40311.parquet', 'edge_reasoning_train_39334_44366.parquet', 'edge_reasoning_train_3957_21412.parquet',
    'edge_reasoning_train_39598_41432.parquet', 'edge_reasoning_train_40600_42920.parquet', 'edge_reasoning_train_42281_44789.parquet', 'edge_reasoning_train_4258_7308.parquet',
    'edge_reasoning_train_42675_51563.parquet', 'edge_reasoning_train_44146_46128.parquet', 'edge_reasoning_train_45449_47215.parquet', 'edge_reasoning_train_46221_56229.parquet',
    'edge_reasoning_train_46859_48306.parquet', 'edge_reasoning_train_47284_49449.parquet', 'edge_reasoning_train_48902_50508.parquet', 'edge_reasoning_train_49968_51579.parquet',
    'edge_reasoning_train_50560_52737.parquet', 'edge_reasoning_train_51912_53850.parquet', 'edge_reasoning_train_53279_55321.parquet', 'edge_reasoning_train_54232_56893.parquet',
    'edge_reasoning_train_5428_6728.parquet', 'edge_reasoning_train_54893_63349.parquet', 'edge_reasoning_train_55550_58070.parquet', 'edge_reasoning_train_57285_59152.parquet',
    'edge_reasoning_train_58202_60283.parquet', 'edge_reasoning_train_58918_65481.parquet', 'edge_reasoning_train_59664_61365.parquet', 'edge_reasoning_train_5_1893.parquet',
    'edge_reasoning_train_60762_62496.parquet', 'edge_reasoning_train_61461_63907.parquet', 'edge_reasoning_train_6281_8375.parquet', 'edge_reasoning_train_62833_65702.parquet',
    'edge_reasoning_train_63922_67425.parquet', 'edge_reasoning_train_64786_68022.parquet', 'edge_reasoning_train_65484_81645.parquet', 'edge_reasoning_train_66169_69231.parquet',
    'edge_reasoning_train_66753_70081.parquet', 'edge_reasoning_train_67897_70610.parquet', 'edge_reasoning_train_69186_70068.parquet', 'edge_reasoning_train_69437_71621.parquet',
    'edge_reasoning_train_71117_72627.parquet', 'edge_reasoning_train_72120_73618.parquet', 'edge_reasoning_train_73060_74633.parquet', 'edge_reasoning_train_73790_75770.parquet',
    'edge_reasoning_train_74972_76817.parquet', 'edge_reasoning_train_75970_77839.parquet', 'edge_reasoning_train_76951_78843.parquet', 'edge_reasoning_train_77924_81239.parquet',
    'edge_reasoning_train_7826_9477.parquet', 'edge_reasoning_train_80625_83628.parquet', 'edge_reasoning_train_81058_88838.parquet', 'edge_reasoning_train_81886_98465.parquet',
    'edge_reasoning_train_83031_84630.parquet', 'edge_reasoning_train_84144_85703.parquet', 'edge_reasoning_train_84946_86725.parquet', 'edge_reasoning_train_85728_87696.parquet',
    'edge_reasoning_train_86715_88685.parquet', 'edge_reasoning_train_87784_89670.parquet', 'edge_reasoning_train_8826_10587.parquet', 'edge_reasoning_train_89152_91008.parquet',
    'edge_reasoning_train_90413_92466.parquet', 'edge_reasoning_train_90419_108950.parquet', 'edge_reasoning_train_91583_93595.parquet', 'edge_reasoning_train_92487_94587.parquet',
    'edge_reasoning_train_93981_95619.parquet', 'edge_reasoning_train_94740_96673.parquet', 'edge_reasoning_train_96043_97732.parquet', 'edge_reasoning_train_96777_98947.parquet',
    'edge_reasoning_train_97975_99957.parquet', 'edge_reasoning_train_98239_214817.parquet', 'edge_reasoning_train_99013_100966.parquet',
)


def sources() -> list[RlDataSource[CurationRecipe]]:
    """The declared Edge WebSearch preparation-planning cohort (TaskForge-authored)."""
    return [source(EDGE_DATA_FILES)]
