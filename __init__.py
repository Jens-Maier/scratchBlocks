bl_info = {
    "name": "scratchBlocks",
    "author": "Jens Maier",
    "version": (0, 0, 1),
    "blender": (4, 5, 14),
    "location": "",
    "description": "Scratch blocks",
    "warning": "",
    "doc_url": "",
    "category": "3D View",
}

#import importlib
#import mathutils
import bpy
from . import scratch_data
from . import scratch_ui


classes = (
    scratch_ui.SCRATCH_OT_move_blocks_operator,
    scratch_ui.SCRATCH_OT_call_add_menu,
    scratch_ui.SCRATCH_OT_add_block_operator,
    scratch_ui.SCRATCH_OT_evaluate_operator,
    scratch_ui.SCRATCH_MT_add_menu,
    scratch_ui.SCRATCH_OT_new_object_graph,
    scratch_ui.SCRATCH_OT_save_object_graph, 
    scratch_ui.SCRATCH_OT_load_object_graph, 
    scratch_ui.SCRATCH_PT_panel,
)

addon_keymaps = []

def register_keymaps():
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name="Image", space_type='IMAGE_EDITOR')
        
        # 1. Primary shortcut: Ctrl + A
        kmi_ctrl = km.keymap_items.new("scratch.call_add_menu", type='A', value='PRESS', ctrl=True)
        addon_keymaps.append((km, kmi_ctrl))
        
        # 2. Secondary shortcut: Shift + A (Blender muscle-memory alias)
        kmi_shift = km.keymap_items.new("scratch.call_add_menu", type='A', value='PRESS', shift=True)
        addon_keymaps.append((km, kmi_shift))

def unregister_keymaps():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

def register():
    print("register() scratchBlocks")
    for c in classes:
        bpy.utils.register_class(c)

    # Register object property for per-object graph mapping
    bpy.types.Object.scratch_graph_name = bpy.props.StringProperty(
        name="Scratch Graph", 
        description="Name of the scratch graph assigned to this object", 
        default="",
        update=scratch_data.update_scratch_graph_property
    )

    # Register selection handler for auto-loading
    if scratch_ui.load_graph_on_object_change not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(scratch_ui.load_graph_on_object_change)
        
    scratch_ui.register_draw_handler()
    register_keymaps()
    bpy.types.IMAGE_HT_header.append(scratch_ui.draw_header_menu)

def unregister():
    print("unregister() scratchBlocks")
    bpy.types.IMAGE_HT_header.remove(scratch_ui.draw_header_menu)
    unregister_keymaps()
    scratch_ui.unregister_draw_handler()

    if scratch_ui.load_graph_on_object_change in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(scratch_ui.load_graph_on_object_change)

    del bpy.types.Object.scratch_graph_name
    
    for c in reversed(classes):
        bpy.utils.unregister_class(c)