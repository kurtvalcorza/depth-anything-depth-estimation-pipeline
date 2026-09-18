"""Depth-labelled image dataset contract for adapting the relative-depth estimator: the pinned DIODE
validation sample, validation, seeded scan-level splitting, BYOD loaders and CSV export.

The default dataset is **real** and carries metric ground truth: 40 views from the DIODE validation release
(Vasiljevic et al., 2019; CC BY 4.0) — the first two views in file-name order of every one of its 20 scans
(10 indoor, 10 outdoor) — chosen a priori on 2026-09-19 and pinned here per file (RGB PNG, depth `.npy` in
metres, validity-mask `.npy`) by byte size and SHA-256 as served by the Marigold evaluation mirror of DIODE on
the Hugging Face Hub at an immutable commit. Every file is fetched at run time and refused on any byte-size or
SHA-256 mismatch; the repository redistributes none of them. Views of one scan share a scene, so the sample is
split **by scan**, never by view.

A record is ``{id, image, depth, mask}``: a PIL image (or a path to one), a float32 H x W array of metric
depth in metres (or a path to a `.npy`) and a boolean H x W validity mask (or a path; missing means
``depth > 0``). ``domain`` / ``scene`` / ``scan`` are optional provenance keys; ``scan`` (or ``group``) is the
split unit when present.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .pipeline import MAX_ASPECT_RATIO, MAX_IMAGE_SIDE, MIN_IMAGE_SIDE, MODEL_ID

CORPUS_NAME = "DIODE validation views (indoor + outdoor) with metric depth"
CORPUS_RELEASE = (
    "DIODE validation release as served by the Marigold evaluation mirror "
    "(obukhovai/marigold_depth_eval @ 30c5b061, Hugging Face Hub); 40 views pinned 2026-09-19"
)
CORPUS_COMMIT = "30c5b061d863e383e3ea9fa14555737a199d7ad9"
CORPUS_BASE_URL = (
    f"https://huggingface.co/datasets/obukhovai/marigold_depth_eval/resolve/{CORPUS_COMMIT}/diode/"
)
CORPUS_LICENSE = "CC BY 4.0 (DIODE: A Dense Indoor and Outdoor DEpth Dataset, Vasiljevic et al., 2019)"
CORPUS_BYTES = 312_448_846
DOMAINS = ("indoors", "outdoor")
FILE_SUFFIXES = (".png", "_depth.npy", "_depth_mask.npy")

# (record id, domain, scene, scan, file stem, then bytes + sha256 of the png, depth .npy and mask .npy)
SAMPLE_RECORDS: tuple[tuple[str, str, str, str, str, int, str, int, str, int, str], ...] = (
    (
        "view-000",
        "indoors",
        "scene_00019",
        "scan_00183",
        "00019_00183_indoors_000_010",
        1172425,
        "71f9713e01a4d9899184b63f293446e62953ee2e526d81e5da8f101f2400ae46",
        3145856,
        "a2bbbf55765ffec98a75d3a78f9925b28bfe476080d3e8eae86c645db4ac2b15",
        3145856,
        "e9b5f1cce7c4e8e03a94b006f81ef72d963bfb9b7187f8208f9e92067db823fa",
    ),
    (
        "view-001",
        "indoors",
        "scene_00019",
        "scan_00183",
        "00019_00183_indoors_000_040",
        1102623,
        "ba05f76208099f78dcf483908c7de07ad4d89784ae694aefdbb5a82dc2bfedf1",
        3145856,
        "c72d44e851b649ad787de4d0a21e5c115660779d4d1f8e054e8af201748c2c39",
        3145856,
        "304216b2ce8825e529d836e7ebe36dd80090c19b2e620029a158f1dd42a3e42c",
    ),
    (
        "view-002",
        "indoors",
        "scene_00020",
        "scan_00184",
        "00020_00184_indoors_050_000",
        1142437,
        "2641d27768b547e53d4d1dac922b72cb31a0413762bd78b36b898386c150d40e",
        3145856,
        "d4943f5e28555bfd59e609e7116539f2efd68c8a9378dcf696dca0e078f62004",
        3145856,
        "789dabf2482fa3e3d4bab67b81d49518e1b618f7ba55b82edae10ac8d18e4855",
    ),
    (
        "view-003",
        "indoors",
        "scene_00020",
        "scan_00184",
        "00020_00184_indoors_050_020",
        1017065,
        "a44432a3ba6bab0f6b4bb4326df5db4347a6eb577f74f83141c59aae6810b6a4",
        3145856,
        "d5f628673cfcbff12082cfe2dee75c71d8521560486946369dca90b369aa9b5c",
        3145856,
        "06077d743933e907c17b9e3e2b5146987e87f1b0a2122c784ada4f55a8f8b3f4",
    ),
    (
        "view-004",
        "indoors",
        "scene_00020",
        "scan_00185",
        "00020_00185_indoors_000_000",
        1208304,
        "1bf8317ae6d1ef6c412ff233bda2944568aa7e9169606ed5f694d9c2bc1032f9",
        3145856,
        "6de0b81413613d2abecea29e327220a6d94d54d5ab1bccca786a9720993a4e30",
        3145856,
        "afc7dad580ec7d8f41884032f0c97a8bcee31e76d22cc7c2ab988b28daef4dd0",
    ),
    (
        "view-005",
        "indoors",
        "scene_00020",
        "scan_00185",
        "00020_00185_indoors_000_020",
        1213267,
        "a632cec9d93174aaf8adab73a36e9afc12cb08c99c551ce02e97315de2a288d7",
        3145856,
        "f17ee07bb9b513770b0a98f53c53440957de3b04b162a23991cc92a64cc678ce",
        3145856,
        "a302df41411a3d947b1a3378f6eb64df763726e84470acd9c47fcaa02f7f4d21",
    ),
    (
        "view-006",
        "indoors",
        "scene_00020",
        "scan_00186",
        "00020_00186_indoors_000_000",
        1194403,
        "2478882bbac47733a075b3871d178e4c27ba432d0f6d2af1341265a8eecf1369",
        3145856,
        "b4c7533123e6c7204e6b67d4494af0e59bb454b476e552539c2b058509fc2f06",
        3145856,
        "dbf9d3e124e71fe633411f6e54d58f712095044dd152e3f8d18cbd5c75472c42",
    ),
    (
        "view-007",
        "indoors",
        "scene_00020",
        "scan_00186",
        "00020_00186_indoors_020_000",
        1171589,
        "5a7ffa68320043e62c35319b8b2e971bd29331efef181e92ca04cac17a9f4605",
        3145856,
        "7b488f809c23c62594406d1e8fafb4857d7da7d6c98571cc0e3a4a10b76bbd71",
        3145856,
        "403d0dbaa52767792e69a4471c03a6c0dca174dd517dc8c567dd05851c4b3528",
    ),
    (
        "view-008",
        "indoors",
        "scene_00020",
        "scan_00187",
        "00020_00187_indoors_000_000",
        1258618,
        "16bf17dce5d4025536bada42024cd756c7e6f6ee5012a7e7c606039c7e7028db",
        3145856,
        "77068b8f5e9faa2e6cc89f381720873a36b5631643f3df4cff40d8f59d10ed28",
        3145856,
        "28d3e14c26011438587539d9e5c4e9fecbd631f02727fd35ac5f533cbabfb8fb",
    ),
    (
        "view-009",
        "indoors",
        "scene_00020",
        "scan_00187",
        "00020_00187_indoors_000_020",
        1202574,
        "e241249dadc41376befa417e33165412a834841334cee9e950729eaecb1243ce",
        3145856,
        "3ee01eb76928b1cdc49e81a9a7af29797fb4d25c5409754314972da29ed9f9dd",
        3145856,
        "1ec90f9292cc265e1bd8865757fc09cd119ba4ec9c6cb2863803823b46cc994f",
    ),
    (
        "view-010",
        "indoors",
        "scene_00021",
        "scan_00188",
        "00021_00188_indoors_090_000",
        981100,
        "1040b93aa913ad042231e9b9809293745b02c24c90c0cba706a44c77b7847f56",
        3145856,
        "b4a9073b489bff9d472576592185733ee7a709a7aa04445f0f43356ee333e248",
        3145856,
        "940df9212e7cde2ba9225848f57f3aa8d15d6c9dfe2f63bbf3aae00e5c09c282",
    ),
    (
        "view-011",
        "indoors",
        "scene_00021",
        "scan_00188",
        "00021_00188_indoors_090_020",
        1057542,
        "ffe1f5390dca8b043d672a568d28b4f3409276dce13da76f3d18b16e56bb85a9",
        3145856,
        "d7a5487729ca00dc8ce1f0c08006246143e34b79e8da211b0b4ce81a21db23aa",
        3145856,
        "2922ae237dc3d890ed80d0cd8d704c61728ced3383cf249c24d2c36246212853",
    ),
    (
        "view-012",
        "indoors",
        "scene_00021",
        "scan_00189",
        "00021_00189_indoors_000_000",
        1036343,
        "fed8f8be2ea611b2d1ce2b75440f029d6b786cef7218bace273e96dc79acf60b",
        3145856,
        "e71a79b0a7a22fb6a979543809abb2bd7f906487a7b2e9078d4d93cfc562eed7",
        3145856,
        "4b053fdf65b2123bdcadb43e3629521a8eacce52d7fcd820ccddfb8f3ae39663",
    ),
    (
        "view-013",
        "indoors",
        "scene_00021",
        "scan_00189",
        "00021_00189_indoors_000_020",
        1075564,
        "8dd7d9980c7c79ea74d5438154cbe83fc6e245189b9d36406fc1509d8b7b32e9",
        3145856,
        "9b5f1be74b6fc3901fffe7dc38be3d3a2662ad92a565e9b1e263d2b5784bbb94",
        3145856,
        "625221f96afa15cf03e409dcd04910c793f2a187d85b822f834fead3ae601da9",
    ),
    (
        "view-014",
        "indoors",
        "scene_00021",
        "scan_00190",
        "00021_00190_indoors_050_000",
        1000911,
        "4d90cc37cf28095d1aa45668f6f6457eca7f28560a0f62b640b0802a1f5394eb",
        3145856,
        "c939b634fb09d8a08a111bb6739e4298245dea12171becff202d900b72284fc7",
        3145856,
        "39e2709c2a01a290103bf55f48d4674ff08e213ac7f42afe7cc0e62cc43bad68",
    ),
    (
        "view-015",
        "indoors",
        "scene_00021",
        "scan_00190",
        "00021_00190_indoors_050_030",
        996884,
        "851a7f44cd3ebfca766a3bd25a818a000277f2752f16c0e1bac0655ba77f6168",
        3145856,
        "1d11b9096ab6eeca053307d39eb2a7c3db9d1cc529b5c243421c111eca6ea0a1",
        6291584,
        "8192c4b01dd2c6776cc03f68e9a61f792c9149e27ea7959e362c224193a85349",
    ),
    (
        "view-016",
        "indoors",
        "scene_00021",
        "scan_00191",
        "00021_00191_indoors_000_040",
        1057445,
        "2a7fafdbdfbaf1bbb9363b32b5337a86f391ebdd8cb23a0227d5e6ec3bddc11d",
        3145856,
        "64609dc44db5769caae0aa5b5a29a92a380ff5aaf9c0173c172a7dda0cd8f42a",
        3145856,
        "7c2ee20e3128526b51067419562fba26d25ba734d2a92a31467e9b42d190f0a8",
    ),
    (
        "view-017",
        "indoors",
        "scene_00021",
        "scan_00191",
        "00021_00191_indoors_010_030",
        1053502,
        "3a686ca893ba52f2780370737b762080b2cf2ce7201c4311a9f31ed29595f92f",
        3145856,
        "ece8e9acdeb8601bc111d70b6260d0ea92cea926166dc5b0c1d2bde3f5406b8e",
        3145856,
        "bdf32493a0ad439062821d9f9f3f3a966454c431d2c4bd4b5d22be89bceb494a",
    ),
    (
        "view-018",
        "indoors",
        "scene_00021",
        "scan_00192",
        "00021_00192_indoors_000_000",
        1061846,
        "28c6d3acbf088975eddc38ab6e5d4162b7f93fba8f5459d0f833c2d30fe89530",
        3145856,
        "ebbec603a9625ee58cfd861770402c1cd9fafeb5b6126e6a9a7628941bddefce",
        3145856,
        "0eeeadfb8976b9a77f3ce4351401bec8b1e61ba0a4b2a8397e7328dae738cde5",
    ),
    (
        "view-019",
        "indoors",
        "scene_00021",
        "scan_00192",
        "00021_00192_indoors_000_020",
        1057090,
        "fe69baf50e4801abd3b5f9b2dacd70e33a0ed8c8f4a2d975c980128bf1802e74",
        3145856,
        "2aba7a7f71b5dbd799a42609aab58ae2b6ad7ca7521dc05db5288b05010b585c",
        3145856,
        "1263a8da470fcdaad3be5d70034adeac2f2d7b8aa862fa1f962ab908e5a2184f",
    ),
    (
        "view-020",
        "outdoor",
        "scene_00022",
        "scan_00193",
        "00022_00193_outdoor_000_000",
        1260962,
        "f513f607fbdcdbe3545cfc67d90828d81619be157689c1db26826559c1cae0a6",
        3145856,
        "a145d0f58dfabc71e59bb5831d42b11989d2be3d8bb6ef16ca16449f77907b41",
        3145856,
        "c06249a9c2d6b204fba67219db98be321f5fa39707b67d3b2e3880fe6a707c55",
    ),
    (
        "view-021",
        "outdoor",
        "scene_00022",
        "scan_00193",
        "00022_00193_outdoor_000_020",
        1387923,
        "2fa8096f650f9bb7327cffa8079d471feaffd11b1a37b3133c456d4bb89356de",
        3145856,
        "59513f5e83be4ab75dd8a996704fcbb32f1a4b1143c874654d6106781b7386b7",
        3145856,
        "1ec6d6f79ba218282caf7ae1220b0fa19d06e56e09458b0a85e0825a91545049",
    ),
    (
        "view-022",
        "outdoor",
        "scene_00022",
        "scan_00194",
        "00022_00194_outdoor_000_000",
        1386881,
        "01786324ca6c763a67e32adec512d83d32c295e3243d23fce9fa1b56f8031eb3",
        3145856,
        "84024306218657d9eea51c2abd54044e6ee6293d0193a7e185ed98b1f302b91e",
        6291584,
        "7064a6740d486b03a771a14d9801807dfd6dc2e48ac2c0f42417e755a63f067a",
    ),
    (
        "view-023",
        "outdoor",
        "scene_00022",
        "scan_00194",
        "00022_00194_outdoor_000_020",
        1550903,
        "3e759d88c101e158a81eecf16b87c6e96b4ecd3d5d6e912a46af789a1cfd2f1a",
        3145856,
        "1aa6fd2ef08489b8d0a8547a2fc6093936f524df56967d797363baad7c6f0fff",
        3145856,
        "933093e8d49a222a2329f89d4adb82a8e4eb9e8a72079d885e8eebf97f31f801",
    ),
    (
        "view-024",
        "outdoor",
        "scene_00022",
        "scan_00195",
        "00022_00195_outdoor_000_000",
        1354956,
        "485ae591b4e9197422d23834e658d3b7c0b2a9a8a1f4d190dc4e8285abcf4b58",
        3145856,
        "a0eb3b67dbdb6c7a541bc787f8b2c83c927b510477b83ea8eb47e0c0b64520fc",
        3145856,
        "1425f824f4bd3bfd4bc39f1ef6cff65c24578a79a6a5e56b1813fd68e6405daa",
    ),
    (
        "view-025",
        "outdoor",
        "scene_00022",
        "scan_00195",
        "00022_00195_outdoor_000_020",
        1312039,
        "b505cf1a9e05be9caecc3141b3f6628e2d52aef7bd6768e0495d1fe68f022cc7",
        3145856,
        "00ca098eb81e758870344c0ea0671c15a75964ebc9afa82e5c842e9edf465b1b",
        3145856,
        "9c3b3f29d5c27d8ab1a899528fd98d48a608422f5d6878440c56e9452209082d",
    ),
    (
        "view-026",
        "outdoor",
        "scene_00022",
        "scan_00196",
        "00022_00196_outdoor_000_010",
        1548233,
        "aa7cc5973a301d69420c74169d70d850d206adab453a1677c81ad3df8e456c01",
        3145856,
        "d415a6bb86593c6bb35a6457bbc13cd451206907e6225ca7d37f0171a5a19dc9",
        3145856,
        "63c640cb2c1d438832cacb039a307141e1a5e426d2f3fd2961e2b4e566215e47",
    ),
    (
        "view-027",
        "outdoor",
        "scene_00022",
        "scan_00196",
        "00022_00196_outdoor_000_030",
        1581362,
        "64683db027716e6d5564c8962e2ef819be8e992a7514c5f0f3f42fa056230504",
        3145856,
        "bee31f7ca3020b9415e3debff8c219ee3502b104b1844d438a51d84a199979e9",
        3145856,
        "1aec43f5993a614a4f2b6d7577739615fc7780f13a49fbf899ee24753132e5a0",
    ),
    (
        "view-028",
        "outdoor",
        "scene_00022",
        "scan_00197",
        "00022_00197_outdoor_000_000",
        1397626,
        "7b491f25a7a26e435ef5feb4d90ce5929954a2473e34f76bd70cc2d9a81251c3",
        3145856,
        "5045c2038970c4c282ffccb141c3afef9027a71523d4796189c59a544abc8847",
        3145856,
        "a80a8aa9ecadbef495966bc08935e8e02035c382c8ff8d75075646ee14e00191",
    ),
    (
        "view-029",
        "outdoor",
        "scene_00022",
        "scan_00197",
        "00022_00197_outdoor_010_010",
        1427912,
        "ab669c4bf03ca844c125cb3e6d963364f5f14b36d0af9288130d85a174340205",
        3145856,
        "cd22ceb5a19427d36e1199e167237b0626b468947aa6ec21ef2889b3f687a42e",
        3145856,
        "69912b6858ed773c453b90993ef0c886b69c3bb3ee43644e54682b1e4f11f4a5",
    ),
    (
        "view-030",
        "outdoor",
        "scene_00023",
        "scan_00198",
        "00023_00198_outdoor_000_020",
        1196280,
        "8e058762aeb6d508cc4a453bb51bb4195596644fdce856199e8ff47289154163",
        3145856,
        "07f0d1b82b79e37df0afa504b15061a1d801ffb7143da44d98c7e4573c126b30",
        3145856,
        "fdcca8151892640e098085128f1e34786f79c9ebd8f01c7f14f67aeff5ee529a",
    ),
    (
        "view-031",
        "outdoor",
        "scene_00023",
        "scan_00198",
        "00023_00198_outdoor_070_020",
        1153475,
        "2c6ba4a04b24415855c6d56f9e424e752682c9598ed0a6961a2c095c0fde80e7",
        3145856,
        "0e1cef73cd8aa9611eee6a36395bda320bdbc5608d9431bcce24e59a1e0908d1",
        6291584,
        "4571c9f2b44300fbc1a4152b669720d367860f01827cc5512fd939f24965b7a6",
    ),
    (
        "view-032",
        "outdoor",
        "scene_00023",
        "scan_00199",
        "00023_00199_outdoor_000_020",
        1443825,
        "065fdaf432f60cab7d4d85f77f6e8a8da2d8d7f6b618173e61b71c4db79b1b14",
        3145856,
        "734166fea6643e708cda8075833cffb604a288f4b8790e2c5d758deacb7867d9",
        3145856,
        "ee02105445049bd21b66b642dae6da7992d410b4a5f7c17397dfef192e12b246",
    ),
    (
        "view-033",
        "outdoor",
        "scene_00023",
        "scan_00199",
        "00023_00199_outdoor_010_000",
        1401671,
        "27ac9a575d259ce2511a02b9c5265e39dcbebe4afd39e9f8b64a296ef6ea1095",
        3145856,
        "303990f57fcd277a43df832f1f03ef42a78008f99d0af024fe3e4ed53c977db4",
        3145856,
        "94c855bbefce29e551ceeb0b9971fcd6fa20adee845ab212147d694f6362d813",
    ),
    (
        "view-034",
        "outdoor",
        "scene_00023",
        "scan_00200",
        "00023_00200_outdoor_000_010",
        1103637,
        "3d1c0f779183fe7eb383e05237a5462dc18fd790f35a80e90787e94d0e43c1c7",
        3145856,
        "b515576843d3ad6bd718d8e578c7e3bd128f82bcdadc51d2855ce77f65c63279",
        3145856,
        "23c79f29ae3ea4c3cc4f4e4cbfc4316b5e86a8fb661d1b2cfb2ddaf88f3e12de",
    ),
    (
        "view-035",
        "outdoor",
        "scene_00023",
        "scan_00200",
        "00023_00200_outdoor_000_050",
        1225980,
        "6046d35e5454ceac3691c5086d262431260b9839194be58fd9015415f21b5dba",
        3145856,
        "672aa46a95e17433425bd0510f502eebb637fe494db3afe41c36b818ab3d7334",
        3145856,
        "1434e085120fca95356d956046c0b55120d0b4a32a5ec14394d67104f38eb1eb",
    ),
    (
        "view-036",
        "outdoor",
        "scene_00024",
        "scan_00201",
        "00024_00201_outdoor_000_000",
        1916342,
        "2fd43a35faee39cdcf8b08b1f23915c8aabae57601e1c4eb90ac98c2726f6726",
        3145856,
        "7585ead8e9e351d703220c1fb4aa1a070874e0afcdb676646e8db7e90c57d46a",
        3145856,
        "4e6f8a5647539f94a7980255bb7c0be7dfdb7fe30395c3fccfdd47fcbddcb734",
    ),
    (
        "view-037",
        "outdoor",
        "scene_00024",
        "scan_00201",
        "00024_00201_outdoor_000_020",
        1842876,
        "08c4b8342eea57a08d4380108ea14f342b43af3ef10fd7d1f9ee28dac1a2af6b",
        3145856,
        "c15fbde3764e85cb2c2b88a02562101447141b8ed63f0a70fa6da0c8a22a69a1",
        3145856,
        "5a92fbe940ba37d595197c9caa012526ec6b2f32ebceb5ffadb3b98ecafb80c3",
    ),
    (
        "view-038",
        "outdoor",
        "scene_00024",
        "scan_00202",
        "00024_00202_outdoor_000_030",
        1911126,
        "29640b6d05b7f6f310c31a9a26e160518f679439bf619983dbe0b5fd1afb39aa",
        3145856,
        "a85149326d869f091b5611e7af0dfe0b0dad770bd8eb92f21bf4be2c76a65887",
        3145856,
        "f138b3fc58846961bb7193825be86a427d7689cf1f04b23a9eb830988eb93d84",
    ),
    (
        "view-039",
        "outdoor",
        "scene_00024",
        "scan_00202",
        "00024_00202_outdoor_070_010",
        1877641,
        "e6c78b21fdd07ae5feeb21beb3c5f391522f041c9ddb55ba53a47cfd5994e434",
        3145856,
        "942e5fb2d5e63738fbc09c978d6f8ac878f50313c99bbd73ffc9239d0f44f109",
        3145856,
        "37ddc9e81d1c9f6abee51157213f4a8d5579ad4994bf5982220a8dc023dcd05a",
    ),
)

DEFAULT_CACHE_DIR = Path("weights") / "diode-sample"  # working-directory-relative, like the notebook
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 6, "validation": 2, "test": 2}  # scans per domain; 2 domains x 2 views -> 24 / 8 / 8
MIN_RECORDS = 4
MAX_RECORDS = 2_000
MIN_VALID_FRACTION = 0.05  # a depth map must label at least this fraction of its pixels
MAX_DEPTH_M = 1_000.0
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_url(domain: str, scene: str, scan: str, stem: str, suffix: str) -> str:
    return f"{CORPUS_BASE_URL}{domain}/{scene}/{scan}/{stem}{suffix}"


def _pins(record: tuple) -> dict[str, tuple[int, str]]:
    return {
        ".png": (record[5], record[6]),
        "_depth.npy": (record[7], record[8]),
        "_depth_mask.npy": (record[9], record[10]),
    }


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, dict[str, bytes]]:
    """Return every pinned file (bytes keyed by record id, then suffix) from the cache or the Hub mirror."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, bytes]] = {}
    for record in SAMPLE_RECORDS:
        rid, domain, scene, scan, stem = record[:5]
        out[rid] = {}
        for suffix, (size, digest) in _pins(record).items():
            local = cache / f"{stem}{suffix}"
            data = local.read_bytes() if local.is_file() else b""
            if len(data) != size or _sha256_bytes(data) != digest:
                url = file_url(domain, scene, scan, stem, suffix)
                if fetcher is not None:
                    data = fetcher(url)
                else:
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "dimer-depth-anything-tutorial/1.0"}
                    )
                    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 (pinned https URL)
                        data = response.read()
                if len(data) != size or _sha256_bytes(data) != digest:
                    raise ValueError(
                        f"{rid} ({stem}{suffix}): fetched {len(data)} bytes with sha256 "
                        f"{_sha256_bytes(data)[:16]}…, pinned {size} / {digest[:16]}…"
                    )
                local.write_bytes(data)
            out[rid][suffix] = data
    return out


def _load_npy(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def read_corpus(files: Mapping[str, Mapping[str, bytes]]) -> list[dict[str, Any]]:
    """Decode the verified files into `{id, image, depth, mask}` records with their provenance."""
    out = []
    for record in SAMPLE_RECORDS:
        rid, domain, scene, scan, stem = record[:5]
        if rid not in files:
            raise ValueError(f"corpus is missing {rid}")
        image = Image.open(io.BytesIO(files[rid][".png"]))
        image.load()
        depth = _load_npy(files[rid]["_depth.npy"])
        depth = depth[..., 0] if depth.ndim == 3 else depth
        mask = _load_npy(files[rid]["_depth_mask.npy"]) > 0
        out.append(
            {
                "id": rid,
                "image": image.convert("RGB"),
                "depth": np.ascontiguousarray(depth, dtype=np.float32),
                "mask": np.ascontiguousarray(mask, dtype=bool),
                "domain": domain,
                "scene": scene,
                "scan": scan,
                "source_stem": stem,
                "source_url": file_url(domain, scene, scan, stem, ".png"),
            }
        )
    return out


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded draw of whole scans per domain: `sizes` = scans per domain for train / validation / test."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    rng = random.Random(seed)
    by_scan: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        by_scan.setdefault((str(record.get("domain", "")), str(record["scan"])), []).append(dict(record))
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in sizes}
    for domain in sorted({d for d, _ in by_scan}):
        scans = sorted(s for d, s in by_scan if d == domain)
        rng.shuffle(scans)
        needed = sum(sizes.values())
        if len(scans) < needed:
            raise ValueError(f"{domain}: only {len(scans)} scans available, need {needed}")
        cursor = 0
        for name, per_domain in sizes.items():
            for scan in scans[cursor : cursor + per_domain]:
                out[name].extend(by_scan[(domain, scan)])
            cursor += per_domain
    for name in out:
        rng.shuffle(out[name])
        out[name] = [{**r, "id": f"{name}-{i:03d}", "source_id": r["id"]} for i, r in enumerate(out[name])]
    return out


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Any = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus."""
    return build_sample_dataset(
        read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher)), seed=seed, sizes=sizes
    )


def _as_array(value: Any, label_name: str, key: str) -> np.ndarray:
    if isinstance(value, str | Path):
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"{label_name}: {key} file not found: {path}")
        value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{label_name}: {key} must be a numpy array or a path to a .npy file")
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"{label_name}: {key} must be a 2-D H x W array, got shape {value.shape}")
    return value


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label_name = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label_name} must be a mapping with id/image/depth[/mask]")
    for key in ("id", "image", "depth"):
        if key not in record:
            raise ValueError(f"{label_name} is missing {key!r}")
    rid, image = record["id"], record["image"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label_name}: id must match {_ID_RE.pattern}")
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"{label_name}: image file not found: {path}")
        image = Image.open(path)
        image.load()
    if not isinstance(image, Image.Image):
        raise ValueError(f"{label_name}: image must be a PIL.Image.Image or a file path")
    width, height = image.size
    short, long = min(width, height), max(width, height)
    if short < MIN_IMAGE_SIDE or long > MAX_IMAGE_SIDE:
        raise ValueError(
            f"{label_name}: image side outside {MIN_IMAGE_SIDE}..MAX_IMAGE_SIDE={MAX_IMAGE_SIDE} px: "
            f"{image.size}"
        )
    if long / short > MAX_ASPECT_RATIO:
        raise ValueError(
            f"{label_name}: aspect ratio {long / short:.2f} > MAX_ASPECT_RATIO {MAX_ASPECT_RATIO}"
        )
    depth = _as_array(record["depth"], label_name, "depth")
    if depth.shape != (height, width):
        raise ValueError(
            f"{label_name}: depth shape {depth.shape} != image (height, width) {(height, width)}"
        )
    if not np.issubdtype(depth.dtype, np.number):
        raise ValueError(f"{label_name}: depth must be numeric, got {depth.dtype}")
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    if record.get("mask") is None:
        mask = np.isfinite(depth) & (depth > 0)
    else:
        mask = _as_array(record["mask"], label_name, "mask")
        if mask.shape != depth.shape:
            raise ValueError(f"{label_name}: mask shape {mask.shape} != depth shape {depth.shape}")
        mask = np.ascontiguousarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    fraction = float(mask.mean())
    if fraction < MIN_VALID_FRACTION:
        raise ValueError(
            f"{label_name}: only {fraction:.1%} of pixels carry valid depth; "
            f"at least {MIN_VALID_FRACTION:.0%} are required"
        )
    if float(depth[mask].max()) > MAX_DEPTH_M:
        raise ValueError(f"{label_name}: depth exceeds MAX_DEPTH_M={MAX_DEPTH_M} m (metres are expected)")
    item = {"id": rid, "image": image.convert("RGB"), "depth": depth, "mask": mask}
    for key in ("source_id", "domain", "scene", "scan", "group", "source_stem", "source_url"):
        if key in record:
            item[key] = record[key]
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a depth-labelled image dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, depth[, mask]} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = [_check_record(record, index) for index, record in enumerate(records)]
    ids = [r["id"] for r in checked]
    if len(set(ids)) != len(ids):
        duplicate = next(i for i in ids if ids.count(i) > 1)
        raise ValueError(f"duplicate id {duplicate!r}")
    sides = [max(r["image"].size) for r in checked]
    valid = [float(r["mask"].mean()) for r in checked]
    depths = [float(np.median(r["depth"][r["mask"]])) for r in checked]
    domains: dict[str, int] = {}
    for r in checked:
        domains[str(r.get("domain", "unspecified"))] = domains.get(str(r.get("domain", "unspecified")), 0) + 1
    return {
        "records": checked,
        "n_records": len(checked),
        "domain_counts": domains,
        "scans": len({str(r.get("scan", r.get("group", r["id"]))) for r in checked}),
        "image_side": {"min": min(sides), "max": max(sides)},
        "valid_fraction": {"min": round(min(valid), 4), "max": round(max(valid), 4)},
        "median_depth_m": {"min": round(min(depths), 3), "max": round(max(depths), 3)},
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def image_digest(image: Image.Image) -> str:
    """SHA-256 of the decoded RGB pixels (size + bytes), so a re-encoded copy of the same photo matches."""
    rgb = image.convert("RGB")
    return _sha256_bytes(f"{rgb.size[0]}x{rgb.size[1]}:".encode() + rgb.tobytes())


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        [
            r["id"],
            image_digest(r["image"]),
            _sha256_bytes(np.ascontiguousarray(r["depth"], dtype=np.float32).tobytes()),
        ]
        for r in records
    ]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _split_unit(record: Mapping[str, Any]) -> str:
    return str(record.get("scan") or record.get("group") or record.get("source_id") or record["id"])


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no image (by decoded-pixel digest) and no scan appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    scans: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = image_digest(record["image"])
            if key in seen and seen[key] != name:
                raise ValueError(f"image {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
            unit = _split_unit(record)
            if unit in scans and scans[unit] != name:
                raise ValueError(f"scan {unit!r} has views in both {scans[unit]} and {name}")
            scans[unit] = name
    return {name: len(records) for name, records in splits.items()}


def scan_summary(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Scans and domains per split (an observation of what the split unit was)."""
    out: dict[str, Any] = {}
    for name, records in splits.items():
        out[name] = {
            "views": len(records),
            "scans": len({_split_unit(r) for r in records}),
            "domains": {
                d: sum(1 for r in records if str(r.get("domain", "unspecified")) == d)
                for d in sorted({str(r.get("domain", "unspecified")) for r in records})
            },
        }
    return out


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test by split unit (`scan` / `group`, else the
    image itself) after de-duplicating images, so views of one scan never straddle splits."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        key = image_digest(record["image"])
        if key not in seen:
            seen.add(key)
            by_unit.setdefault(_split_unit(record), []).append(record)
    rng = random.Random(seed)
    units = sorted(by_unit)
    rng.shuffle(units)
    n_test = max(1, round(len(units) * test_fraction))
    n_val = round(len(units) * val_fraction)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for name, chosen in (
        ("test", units[:n_test]),
        ("validation", units[n_test : n_test + n_val]),
        ("train", units[n_test + n_val :]),
    ):
        for unit in chosen:
            splits[name].extend(by_unit[unit])
    for part in splits.values():
        rng.shuffle(part)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, image, depth[, mask]}` records from a directory or a zip holding `records.csv` (columns
    `id`, `image`, `depth`, optional `mask` and `group`) beside the files; depth and mask are `.npy` arrays
    (metres; boolean) decoded from the archive, never extracted to disk."""
    source = Path(path)
    if source.is_dir():
        table = (source / "records.csv").read_text(encoding="utf-8")
        loader = lambda name: (source / name).read_bytes()  # noqa: E731
    elif source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        members = {Path(n).name: n for n in archive.namelist()}
        if "records.csv" not in members:
            raise ValueError("BYOD zip must contain records.csv")
        table = archive.read(members["records.csv"]).decode("utf-8")
        loader = lambda name: archive.read(members[name])  # noqa: E731
    else:
        raise ValueError("BYOD datasets must be a directory or a .zip holding records.csv and the files")
    rows = list(csv.DictReader(io.StringIO(table)))
    missing = {"id", "image", "depth"} - set(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"records.csv is missing columns {sorted(missing)}")
    out = []
    for row in rows:
        image = Image.open(io.BytesIO(loader(row["image"])))
        image.load()
        item: dict[str, Any] = {
            "id": row["id"],
            "image": image.convert("RGB"),
            "depth": _load_npy(loader(row["depth"])),
        }
        if row.get("mask"):
            item["mask"] = _load_npy(loader(row["mask"]))
        if row.get("group"):
            item["group"] = row["group"]
        out.append(item)
    return out


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the records table of a split (id, image, depth, mask, group, provenance) as BYOD expects it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "image", "depth", "mask", "group", "domain", "source_url"]
        )
        writer.writeheader()
        for record in records:
            stem = record.get("source_stem") or record["id"]
            writer.writerow(
                {
                    "id": record["id"],
                    "image": f"{stem}.png",
                    "depth": f"{stem}_depth.npy",
                    "mask": f"{stem}_depth_mask.npy",
                    "group": _split_unit(record),
                    "domain": record.get("domain", ""),
                    "source_url": record.get("source_url", ""),
                }
            )
    return out
