import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import blf
from . import scratch_data
from .scratch_logic import run_evaluation_pipeline

from .scratch_data import get_block_shader, canvas_blocks, last_mouse_pos, ScratchBlockData, get_shared_batches, save_current_graph, load_graph

_is_modal_running = False

def get_block_depth(b):
    depth = 0
    curr = getattr(b, 'parent_nest_block', None)
    while curr:
        depth += 1
        curr = getattr(curr, 'parent_nest_block', None)
    return depth

def is_in_dragging_tree(b):
    curr = b
    while curr:
        if getattr(curr, 'is_dragging', False):
            return True
        curr = getattr(curr, 'parent_nest_block', None)
    return False

def get_draw_order(b):
    drag_weight = 1000 if is_in_dragging_tree(b) else 0
    return drag_weight + get_block_depth(b)

def draw_rect_optimized(x, y, w, h, color):
    shader = get_block_shader()
    vertices = ((x, y), (x + w, y), (x, y - h), (x + w, y - h))
    indices = ((0, 1, 2), (2, 1, 3))
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)

def draw_block_callback():
    gpu.state.blend_set('ALPHA')
    shader = get_block_shader()
    shader.bind()

    sorted_blocks = sorted(canvas_blocks, key=get_draw_order)
    for b in sorted_blocks:
        body_batch, outline_batch = get_shared_batches(b.width, b.height)

        # apply matrix translation
        gpu.matrix.push()
        gpu.matrix.translate((b.x, b.y))

        # 1. Main block body
        block_color = getattr(b, 'color', (0.9, 0.6, 0.1, 1.0))
        shader.uniform_float("color", block_color)
        body_batch.draw(shader)
        
        # draw outline
        gpu.state.line_width_set(2.0)
        shader.uniform_float("color", (0.0, 0.0, 0.0, 1.0))
        outline_batch.draw(shader)
        gpu.state.line_width_set(1.0)

        gpu.matrix.pop()
        
        # 2. Block label
        label = getattr(b, 'label', b.block_type)
        if label:
            blf.size(0, 13)
            blf.color(0, 1.0, 1.0, 1.0, 1.0)
            label_y = b.y - (b.height / 2) - 5
            blf.position(0, b.x + 10, label_y, 0)
            blf.draw(0, label)

        # 3. Empty slot grooves for Nesting Blocks
        if getattr(b, 'is_nesting', False):
            slot_rects = b.get_slot_rects()
            for i, (sx, sy, sw, sh) in enumerate(slot_rects):
                if b.slots[i] is None:
                    draw_rect_optimized(sx, sy, sw, sh, (0.15, 0.15, 0.18, 0.85))
                    blf.size(0, 10)
                    blf.color(0, 0.6, 0.6, 0.6, 0.8)
                    blf.position(0, sx + (sw / 2) - 4, sy - (sh / 2) - 4, 0)
                    blf.draw(0, str(i + 1))

        # 4. Multiple Text Input Boxes (z.B. für Vektor X, Y, Z)
        if getattr(b, 'num_text_inputs', 0) > 0 and not getattr(b, 'is_nesting', False):
            rects = b.get_input_rects()
            for i, (bx, by, bw, bh) in enumerate(rects):
                is_active = (b.active_input_index == i)
                box_bg = (1.0, 1.0, 0.82, 1.0) if is_active else (1.0, 1.0, 1.0, 0.95)
                draw_rect_optimized(bx, by, bw, bh, box_bg)
                #draw_rect_outline(bx, by, bw, bh, (0.3, 0.3, 0.3, 1.0), thickness=1.0)
                
                val_str = b.values[i]
                if is_active:
                    pos = max(0, min(b.cursor_pos, len(val_str)))
                    display_text = val_str[:pos] + "|" + val_str[pos:]
                else:
                    display_text = val_str

                blf.size(0, 11)
                blf.color(0, 0.1, 0.1, 0.1, 1.0)
                blf.position(0, bx + 4, by - 17, 0)
                blf.draw(0, display_text)

        # 5. Read-only Text Display Field for Mesh Primitives
        if getattr(b, 'has_display_text', False) and not getattr(b, 'is_nesting', False):
            rects = b.get_display_rect()
            if rects:
                bx, by, bw, bh = rects
                # draw read-only box background
                draw_rect_optimized(bx, by, bw, bh, (0.85, 0.85, 0.85, 0.95))

                display_text = str(b.target_object_name) if b.target_object_name else "(none)"

                blf.size(0, 11)
                blf.color(0, 0.2, 0.2, 0.2, 1.0)
                blf.position(0, bx + 6, by - 17, 0)
                blf.draw(0, display_text)


    gpu.state.blend_set('NONE')

_draw_handle = None

def register_draw_handler():
    global _draw_handle
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceImageEditor.draw_handler_add(
            draw_block_callback, (), 'WINDOW', 'POST_PIXEL'
        )

def unregister_draw_handler():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None


def draw_header_menu(self, context):
    layout = self.layout
    layout.separator()
    layout.menu("SCRATCH_MT_add_menu", text="Add Block", icon='ADD')
    layout.separator()
    layout.operator("scratch.new_object_graph", text="New Graph", icon='ADD')
    obj = context.active_object
    if obj:
        layout.prop(obj, "scratch_graph_name", text="Graph")
    layout.operator("scratch.save_object_graph", text="Save Graph", icon='FILE_TICK')
    layout.operator("scratch.load_object_graph", text="Load Graph", icon='FILE_FOLDER')
    layout.separator()
    layout.operator("scratch.move_blocks", text="Interact Blocks", icon='PLAY')
    layout.operator("scratch.evaluate_graph", text="Evaluate Graph", icon='FILE_SCRIPT')
#def draw_header_menu(self, context):
#    """Optional: Adds an 'Add Block' menu directly in the Image Editor top bar."""
#    layout = self.layout
#    layout.separator()
#    layout.menu("SCRATCH_MT_add_menu", text="Add Block", icon='ADD')
#    layout.operator("scratch.move_blocks", text="Interact Blocks", icon='PLAY')
#    layout.operator("scratch.evaluate_graph", text="Evaluate Graph", icon='FILE_SCRIPT')


# --- Operators ---
class SCRATCH_OT_new_object_graph(bpy.types.Operator):
    bl_idname = "scratch.new_object_graph"
    bl_label = "New Graph"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj:
            self.report({'WARNING'}, "No active object selected")
            return {'CANCELLED'}

        graph_name = f"{obj.name}_graph"
        obj.scratch_graph_name = graph_name
        canvas_blocks.clear()

        initial_block = ScratchBlockData("GEO_OUTPUT", x=200, y=300, label="Group Output", color=(0.85, 0.25, 0.25, 1.0))
        canvas_blocks.append(initial_block)
        save_current_graph(graph_name)
        run_evaluation_pipeline()

        self.report({'INFO'}, f"Created new graph '{graph_name}' for {obj.name}")
        return {'FINISHED'}


class SCRATCH_OT_save_object_graph(bpy.types.Operator):
    bl_idname = "scratch.save_object_graph"
    bl_label = "Save Graph"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or not obj.scratch_graph_name:
            self.report({'WARNING'}, "No graph assigned to active object")
            return {'CANCELLED'}

        save_current_graph(obj.scratch_graph_name)
        self.report({'INFO'}, f"Saved graph '{obj.scratch_graph_name}'")
        return {'FINISHED'}

class SCRATCH_OT_load_object_graph(bpy.types.Operator):
    bl_idname = "scratch.load_object_graph"
    bl_label = "Load Graph"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or not obj.scratch_graph_name:
            self.report({'WARNING'}, "No graph assigned to active object")
            return {'CANCELLED'}

        load_graph(obj.scratch_graph_name)
        run_evaluation_pipeline()
        self.report({'INFO'}, f"Loaded graph '{obj.scratch_graph_name}")
        return {'FINISHED'}

# --- SIDEBAR PANEL ---

class SCRATCH_PT_panel(bpy.types.Panel):
    bl_label = "Scratch Blocks"
    bl_idname = "VIEW3D_PT_scratch_blocks"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Scratch"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        if not obj:
            layout.label(text="Select an object", icon='INFO')
            return

        layout.prop(obj, "scratch_graph_name", text="Graph Name")

        row = layout.row(align=True)
        row.operator("scratch.new_object_graph", text="New", icon='ADD')
        row.operator("scratch.save_object_graph", text="Save", icon='FILE_TICK')
        layout.operator("scratch.load_object_graph", text="Load", icon='FILE_FOLDER')

        layout.separator()
        layout.operator("scratch.evaluate_graph", text="Evaluate Graph", icon='FILE_SCRIPT')

# --- auto-loading handler on object selection ---
_last_active_object = None

@bpy.app.handlers.persistent
def load_graph_on_object_change(scene):
    global _last_active_object_name
    try:
        active_obj = bpy.context.active_object
        current_name = active_obj.scratch_graph_name if (active_obj and hasattr(active_obj, "scratch_graph_name")) else ""

        if active_obj != bpy.context.view_layer.objects.active or current_name != _last_active_object_name:
            _last_active_object_name = current_name
            if active_obj and current_name:
                load_graph(current_name)
                #run_evaluation_pipeline()
    except Exception as e:
        print(f"ScratchBlocks Error in load_graph_on_object_change: {e}")


class SCRATCH_OT_evaluate_operator(bpy.types.Operator):
    bl_idname = "scratch.evaluate_graph"
    bl_label = "Evaluate Scratch Graph"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        run_evaluation_pipeline()
        self.report({'INFO'}, "Scratch Graph evaluated successfully!")
        return {'FINISHED'}


class SCRATCH_OT_move_blocks_operator(bpy.types.Operator):
    bl_idname = "scratch.move_blocks"
    bl_label = "Move Scratch Blocks"
    bl_options = {'REGISTER', 'UNDO'}

    def is_descendant(self, root, target):
        if not root or not target:
            return False
        if root == target:
            return True
        if root.next_block and self.is_descendant(root.next_block, target):
            return True
        if getattr(root, 'is_nesting', False):
            for child in root.slots:
                if child and self.is_descendant(child, target):
                    return True
        return False
        
    def modal(self, context, event):
        global _is_modal_running
        _is_modal_running = True

        if context.area:
            context.area.tag_redraw()

        active_block = next((b for b in canvas_blocks if b.is_dragging), None)
        editing_block = next((b for b in canvas_blocks if getattr(b, 'active_input_index', -1) != -1), None)

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}

        if (event.type == 'A' and event.value == 'PRESS') and (event.ctrl or event.shift):
                    global last_mouse_pos
                    last_mouse_pos[0] = event.mouse_region_x
                    last_mouse_pos[1] = event.mouse_region_y
                    bpy.ops.wm.call_menu('INVOKE_DEFAULT', name="SCRATCH_MT_add_menu")
                    return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            mx = event.mouse_region_x
            my = event.mouse_region_y
        
            # 1. Did the user click inside any text input box?
            clicked_input_block = None
            clicked_idx = -1
            for b in reversed(canvas_blocks):
                c_idx = b.get_clicked_input_index(mx, my)
                if c_idx != -1:
                    clicked_input_block = b
                    clicked_idx = c_idx
                    break
        
            if clicked_input_block:
                for b in canvas_blocks:
                    b.active_input_index = -1
                clicked_input_block.active_input_index = clicked_idx
                clicked_input_block.cursor_pos = len(clicked_input_block.values[clicked_idx])
                return {'RUNNING_MODAL'}
            else:
                # clicked outside text inputs: exit text editing
                was_editing = False
                for b in canvas_blocks:
                    if b.active_input_index != -1:
                        b.active_input_index = -1
                        was_editing = True
                if was_editing:
                    run_evaluation_pipeline()
        
            # 2. Hit test blocks for dragging
            hit_block = None
            sorted_for_hit = sorted(canvas_blocks,key=get_block_depth, reverse=True)
            for b in sorted_for_hit:
                if (b.x <= mx <= b.x + b.width and b.y - b.height <= my <= b.y):
                    hit_block = b
                    break
        
            if hit_block:
                hit_block.unsnap()
                hit_block.is_dragging = True
                self.snap_candidate = None
                self.offset_x = mx - hit_block.x
                self.offset_y = my - hit_block.y
        
                # Evaluate immediately when a block is picked up/unsnapped
                run_evaluation_pipeline()
                return {'RUNNING_MODAL'}

        # --- A. HANDLE KEYSTROKES WHEN EDITING TEXT ---
        if editing_block:
            idx = editing_block.active_input_index
            if 0 <= idx < len(editing_block.values):
                current_val = editing_block.values[idx]
                editing_block.cursor_pos = max(0, min(editing_block.cursor_pos, len(current_val)))

                if event.value == 'PRESS':
                    changed = False
                    if event.type in {'RET', 'NUMPAD_ENTER', 'ESC'}:
                        editing_block.active_input_index = -1
                        run_evaluation_pipeline() # Trigger evaluation on closing text input
                        return {'RUNNING_MODAL'}
                    elif event.type == 'LEFT_ARROW':
                        editing_block.cursor_pos = max(0, editing_block.cursor_pos - 1)
                        return {'RUNNING_MODAL'}
                    elif event.type == 'RIGHT_ARROW':
                        editing_block.cursor_pos = min(len(current_val), editing_block.cursor_pos + 1)
                        return {'RUNNING_MODAL'}
                    elif event.type == 'HOME':
                        editing_block.cursor_pos = 0
                        return {'RUNNING_MODAL'}
                    elif event.type == 'END':
                        editing_block.cursor_pos = len(current_val)
                        return {'RUNNING_MODAL'}
                    elif event.type == 'BACK_SPACE':
                        pos = editing_block.cursor_pos
                        if pos > 0:
                            editing_block.values[idx] = current_val[:pos - 1] + current_val[pos:]
                            editing_block.cursor_pos -= 1
                            changed = True                        
                    elif event.type == 'DEL':
                        pos = editing_block.cursor_pos
                        if pos < len(current_val):
                            editing_block.values[idx] = current_val[:pos] + current_val[pos + 1:]
                            changed = True
                    elif event.ascii and event.ascii.isprintable():
                        if len(current_val) < 10:
                            pos = editing_block.cursor_pos
                            editing_block.values[idx] = current_val[:pos] + event.ascii + current_val[pos:]
                            editing_block.cursor_pos += len(event.ascii)
                            changed = True

                    if changed:
                        #run_evaluation_pipeline()
                        pass
                    return {'RUNNING_MODAL'}

            if event.type not in {'MOUSEMOVE', 'LEFTMOUSE', 'RIGHTMOUSE'}:
                return {'RUNNING_MODAL'}

        
        # --- B. MOUSEMOVE (Snapping & Dragging) ---
        if event.type == 'MOUSEMOVE':
            if active_block:
                ideal_x = event.mouse_region_x - self.offset_x
                ideal_y = event.mouse_region_y - self.offset_y
                threshold = 25
                snap_target = None
                snap_type = None
                snap_slot_idx = -1
                target_x, target_y = ideal_x, ideal_y

                for other in canvas_blocks:
                    if other == active_block or active_block.next_block or self.is_descendant(active_block, other):
                        continue
                    if getattr(other, 'is_nesting', False):
                        slot_rects = other.get_slot_rects()
                        for i, (sx, sy, sw, sh) in enumerate(slot_rects):
                            if other.slots[i] is None:
                                if abs(ideal_x - sx) < threshold and abs(ideal_y - sy) < threshold:
                                    snap_target = other
                                    snap_type = 'SLOT'
                                    snap_slot_idx = i
                                    target_x = sx
                                    target_y = sy
                                    break
                        if snap_target:
                            break

                if not snap_target:
                    for other in canvas_blocks:
                        if other == active_block or other.next_block or other.parent_nest_block or self.is_descendant(active_block, other):
                            continue
                        cand_x = other.x
                        cand_y = other.y - other.height - 4
                        if abs(ideal_x - cand_x) < threshold and abs(ideal_y - cand_y) < threshold:
                            snap_target = other
                            snap_type = 'STACK'
                            target_x = cand_x
                            target_y = cand_y
                            break

                self.snap_candidate = snap_target
                self.snap_type = snap_type
                self.snap_slot_index = snap_slot_idx

                dx = target_x - active_block.x
                dy = target_y - active_block.y
                active_block.update_position(dx, dy)
                #run_evaluation_pipeline()
        

        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if active_block:
                active_block.is_dragging = False
                if self.snap_candidate:
                    if self.snap_type == 'SLOT':
                        parent = self.snap_candidate
                        idx = self.snap_slot_index
                        parent.slots[idx] = active_block
                        active_block.parent_nest_block = parent
                        active_block.parent_slot_index = idx
                        parent.update_layout()
                        parent.update_position(0, 0)
                    elif self.snap_type == 'STACK':
                        active_block.prev_block = self.snap_candidate
                        self.snap_candidate.next_block = active_block
                        target_x = self.snap_candidate.x
                        target_y = self.snap_candidate.y - self.snap_candidate.height - 4
                        dx = target_x - active_block.x
                        dy = target_y - active_block.y
                        active_block.update_position(dx, dy)
                    self.snap_candidate = None
                    self.snap_type = None
                    self.snap_slot_index = -1

                # Evaluate when block is dropped/snapped
                run_evaluation_pipeline()

            return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            if active_block:
                active_block.is_dragging = False
            self.snap_candidate = None
            _is_modal_running = False
            return {'FINISHED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        global _is_modal_running
        if _is_modal_running:
            return {'CANCELLED'}
        
        if context.area.type == 'IMAGE_EDITOR':
            self.offset_x = 0
            self.offset_y = 0
            _is_modal_running = True
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Image editor not found, cannot run operator")
            return {'CANCELLED'}


class SCRATCH_OT_add_block_operator(bpy.types.Operator):
    bl_idname = "scratch.add_block"
    bl_label = "Add Scratch Block"
    bl_options = {'REGISTER', 'UNDO'}

    block_type: bpy.props.StringProperty(name="Block Type", default="NEW_BLOCK")
    label: bpy.props.StringProperty(name="Label", default="")
    has_text_input: bpy.props.BoolProperty(name="Has Input", default=False)
    value: bpy.props.StringProperty(name="Default Value", default="")
    num_text_inputs: bpy.props.IntProperty(name="Num Text Inputs", default=0)
    is_nesting: bpy.props.BoolProperty(name="Is Nesting", default=False)
    num_slots: bpy.props.IntProperty(name="Number of Slots", default=2)
    is_set_label: bpy.props.BoolProperty(name="Is Set Label", default=False)
    is_get_label: bpy.props.BoolProperty(name="Is Get Label", default=False)
    color: bpy.props.FloatVectorProperty(
        name="Color",
        size=4,
        default=(0.9, 0.6, 0.1, 1.0),
        subtype='COLOR'
    )

    def create_block(self, spawn_x, spawn_y):
        display_label = self.label if self.label else self.block_type
        is_nest = bool(self.is_nesting)
        new_block = ScratchBlockData(
            self.block_type,
            x=spawn_x - 80,
            y=spawn_y + 20,
            label=display_label,
            color=tuple(self.color), 
            has_text_input=self.has_text_input,
            value=self.value,
            num_text_inputs=self.num_text_inputs,
            is_nesting=is_nest,
            num_slots=self.num_slots if is_nest else 0,
            is_set_label=self.is_set_label,
            is_get_label=self.is_get_label
        )
        canvas_blocks.append(new_block)
        run_evaluation_pipeline()
        return new_block
    
    def execute(self, context):
        spawn_x = last_mouse_pos[0]
        spawn_y = last_mouse_pos[1]
        self.create_block(spawn_x, spawn_y)
        if context.area:
            context.area.tag_redraw()
        return {'FINISHED'}

    def invoke(self, context, event):
        if context.area.type == 'IMAGE_EDITOR':
            global last_mouse_pos
            last_mouse_pos[0] = event.mouse_region_x
            last_mouse_pos[1] = event.mouse_region_y
            return self.execute(context)
        else:
            self.report({'WARNING'}, "Image editor not found, cannot run operator")
            return {'CANCELLED'}

class SCRATCH_OT_call_add_menu(bpy.types.Operator):
    bl_idname = "scratch.call_add_menu"
    bl_label = "Add Scratch Block Menu"
    bl_options = {'REGISTER'}
    def invoke(self, context, event):
        if context.area.type == 'IMAGE_EDITOR':
            global last_mouse_pos
            last_mouse_pos[0] = event.mouse_region_x
            last_mouse_pos[1] = event.mouse_region_y
            bpy.ops.wm.call_menu('INVOKE_DEFAULT', name="SCRATCH_MT_add_menu")
            return {'FINISHED'}
        return {'PASS_THROUGH'}


class SCRATCH_MT_add_menu(bpy.types.Menu):
    bl_label = "Add Block"
    bl_idname = "SCRATCH_MT_add_menu"

    def draw(self, context):
        layout = self.layout

        # 1. Input Geometry Block
        op = layout.operator("scratch.add_block", text="Input Geometry", icon='OBJECT_DATA')
        op.block_type = "INPUT_GEO"
        op.label = "Input Object:"
        op.num_text_inputs = 0
        op.value = ""
        op.is_nesting = False
        #op.color = (0.20, 0.50, 0.70, 1.0)


        # 2. Mesh Primitive (Cube) - Source (Grün)
        op = layout.operator("scratch.add_block", text="Mesh Primitive (Cube)", icon='MESH_CUBE')
        op.block_type = "MESH_CUBE"
        op.label = "Mesh (Cube)"
        op.num_text_inputs = 0
        op.is_nesting = False
        op.is_set_label = False
        op.is_get_label = False
        #op.color = (0.22, 0.60, 0.35, 1.0)

        # 2b. Mesh Primitive (Cylinder) - Source (Grün)
        op = layout.operator("scratch.add_block", text="Mesh Primitive (Cylinder)", icon='MESH_CYLINDER')
        op.block_type = "MESH_CYLINDER"
        op.label = "Mesh (Cylinder)"
        op.num_text_inputs = 0
        op.is_nesting = False
        op.is_set_label = False
        op.is_get_label = False
        #op.color = (0.22, 0.60, 0.35, 1.0)

        # 2. Vector Math / Vector Input (Blau, mit 3 Float-Inputs für X, Y, Z)
        op = layout.operator("scratch.add_block", text="Vector Math", icon='CON_LOCLIKE')
        op.block_type = "VECTOR_MATH"
        op.label = "Vector"
        op.num_text_inputs = 3
        op.value = "0.0"
        op.is_nesting = False
        op.is_set_label = False
        op.is_get_label = False
        #op.color = (0.25, 0.55, 0.85, 1.0) 

        # --- MATH & EXPRESSION NODES ---
        layout.separator()

        # Math Add (+)
        op = layout.operator("scratch.add_block", text="Math Add (+)", icon='ADD')
        op.block_type = "MATH_ADD"
        op.label = "( [] + [] )"
        op.is_nesting = True
        op.num_slots = 2
        #op.color = (0.30, 0.60, 0.80, 1.0)

        # Math Sine (sin)
        op = layout.operator("scratch.add_block", text="Math Sin", icon='RNA')
        op.block_type = "MATH_SIN"
        op.label = "Sin( [] )"
        op.is_nesting = True
        op.num_slots = 1
        #op.color = (0.90, 0.60, 0.10, 1.0)

        layout.separator()

        # 3. Transform Geometry - Modifier (Orange, 1 Slot für Translation)
        op = layout.operator("scratch.add_block", text="Transform Geometry", icon='OBJECT_DATA')
        op.block_type = "TRANSFORM"
        op.label = "Transform"
        op.num_text_inputs = 0
        op.is_nesting = True
        op.num_slots = 1
        op.is_set_label = False
        op.is_get_label = False
        #op.color = (0.90, 0.60, 0.10, 1.0) 

        # 4. Group Output - Sink (Rot)
        op = layout.operator("scratch.add_block", text="Group Output", icon='EXPORT')
        op.block_type = "GEO_OUTPUT"
        op.label = "Group Output"
        op.num_text_inputs = 0
        op.is_nesting = False
        op.is_set_label = False
        op.is_get_label = False
        #op.color = (0.85, 0.25, 0.25, 1.0) 

        layout.separator()

        # 5. Set Label Block (Koralle)
        op = layout.operator("scratch.add_block", text="Set Label", icon='FILE_TICK')
        op.block_type = "SET_LABEL"
        op.label = "Set:"
        op.num_text_inputs = 1
        op.is_nesting = False
        op.is_set_label = True
        op.is_get_label = False
        op.value = "my_geo"
        #op.color = (0.85, 0.35, 0.20, 1.0)

        # 6. Get Label Block (Hellblau)
        op = layout.operator("scratch.add_block", text="Get Label", icon='EYEDROPPER')
        op.block_type = "GET_LABEL"
        op.label = "Get:"
        op.num_text_inputs = 1
        op.is_nesting = False
        op.is_set_label = False
        op.is_get_label = True
        op.value = "my_geo"
        #op.color = (0.20, 0.50, 0.80, 1.0)
    