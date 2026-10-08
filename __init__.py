"""ComfyUI V3 entrypoint. Importing the package does not import ComfyUI.

SPDX-License-Identifier: Apache-2.0
Adapted from FreeFuse-for-anima __init__.py at
1b924b5dd1f7266fa6e5869331e67a2033e7ec2f.
Modified for FreeFuse-for-sliderlora: lazy imports and Krea2 node registration.
See LICENSE, NOTICE and THIRD_PARTY_NOTICES.md for licensing and attribution.
"""


async def comfy_entrypoint():
    from comfy_api.latest import ComfyExtension
    from .nodes import (Krea2SliderFuseEncode, Krea2SliderFuseSubjects, Krea2SliderFuseSampler,
                        Krea2SliderFuseMaskPreview, Krea2SliderFuseDiagnosticSampler, Krea2SliderFuseDiagnosticSave)

    class SliderFreeFuseExtension(ComfyExtension):
        async def get_node_list(self):
            return [Krea2SliderFuseEncode, Krea2SliderFuseSubjects, Krea2SliderFuseSampler, Krea2SliderFuseMaskPreview,
                    Krea2SliderFuseDiagnosticSampler, Krea2SliderFuseDiagnosticSave]

    return SliderFreeFuseExtension()
