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
    #SCRATCH_GNT_node_tree,
    scratch_ui.SCRATCH_OT_move_blocks_operator,
    scratch_ui.SCRATCH_OT_call_add_menu,
    scratch_ui.SCRATCH_OT_add_block_operator,
    scratch_ui.SCRATCH_OT_evaluate_operator,
    scratch_ui.SCRATCH_MT_add_menu,
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

#def draw_header_menu(self, context):
#    """Optional: Adds an 'Add Block' menu directly in the Image Editor top bar."""
#    layout = self.layout
#    layout.separator()
#    layout.menu("SCRATCH_MT_add_menu", text="Add Block", icon='ADD')
#    layout.operator("scratch.move_blocks", text="Interact Blocks", icon='PLAY')
#    layout.operator("scratch.evaluate_graph", text="Evaluate Graph", icon='FILE_SCRIPT')
#
#def draw_modifier(self, context, layout):
#    row = layout.row()
#    row.prop(self, "node_group", text="Graph")

# for module_name in import_modules:
#     full_name = __name__ + '.' + module_name
#     try:
#         globals()[module_name] = importlib.import_module(full_name)
#         print(f"{full_name} imported successfully")
#     except ImportError as e:
#         print(f"Error importing {full_name}: {e}")
#         raise

#def menu_func(self, context):
#    self.layout.operator(scratch_blocks.SCRATCH_OT_move_blocks_operator.bl_idname)
#    self.layout.operator(scratch_blocks.SCRATCH_OT_add_block_operator.bl_idname)

def register():
    print("register() scratchBlocks")
    for c in classes:
        bpy.utils.register_class(c)

    #bpy.types.VIEW3D_MT_object.append(menu_func)
    scratch_ui.register_draw_handler()
    register_keymaps()
    bpy.types.IMAGE_HT_header.append(scratch_ui.draw_header_menu)

def unregister():
    print("unregister() scratchBlocks")
    bpy.types.IMAGE_HT_header.remove(scratch_ui.draw_header_menu)
    unregister_keymaps()
    scratch_ui.unregister_draw_handler()
    #bpy.types.VIEW3D_MT_object.remove(menu_func)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)