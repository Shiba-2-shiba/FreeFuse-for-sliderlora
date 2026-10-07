"""ComfyUI V3 entrypoint. Importing the package does not import ComfyUI."""


async def comfy_entrypoint():
    from comfy_api.latest import ComfyExtension
    from .nodes import Krea2SliderFuseEncode, Krea2SliderFuseSubjects, Krea2SliderFuseSampler, Krea2SliderFuseMaskPreview

    class SliderFreeFuseExtension(ComfyExtension):
        async def get_node_list(self):
            return [Krea2SliderFuseEncode, Krea2SliderFuseSubjects, Krea2SliderFuseSampler, Krea2SliderFuseMaskPreview]

    return SliderFreeFuseExtension()
