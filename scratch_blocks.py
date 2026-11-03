import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import blf


last_mouse_pos = [100, 200]
_is_modal_running = False

cachedShader = None

def get_block_shader():
    global cachedShader
    if cachedShader is None:
        cachedShader = gpu.shader.from_builtin('UNIFORM_COLOR')
    return cachedShader

_shared_body_batch = None
_shared_outline_batch = None

def get_shared_batches(width, height):
    global _shared_body_batch, _shared_outline_batch
    shader = get_block_shader()

    # Recreate only if dimensions change or uninitialised
    vertices = ((0, 0), (width, 0), (0, -height), (width, -height))
    indices = ((0, 1, 2), (2, 1, 3))
    _shared_body_batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)

    outline_verts = ((0, 0), (width, 0), (width, -height), (0, -height))
    outline_indices = ((0, 1), (1, 2), (2, 3), (3, 0))
    _shared_outline_batch = batch_for_shader(shader, 'LINES', {"pos": outline_verts}, indices=outline_indices)

    return _shared_body_batch, _shared_outline_batch


# data structure for a block
class ScratchBlockData:
    def __init__(self, block_type, x=50, y=100, label=None, color=None, has_text_input=False, value="", num_text_inputs=0, is_nesting=False, num_slots=2, is_set_label=False, is_get_label=False):
        self.block_type = block_type
        self.label = label or block_type
        self.color = color if color is not None else (0.9, 0.6, 0.1, 1.0)

        self.is_set_label = is_set_label
        self.is_get_label = is_get_label
        is_single_input = is_set_label or is_get_label or block_type == "INPUT_GEO"

        # Mehrere Textfelder (z. B. 3 für Vektoren X, Y, Z) oder einzelnes Textfeld
        self.num_text_inputs = 1 if (is_set_label or is_get_label) else num_text_inputs
        if self.num_text_inputs > 0:
            # Werte als Liste initialisieren (z. B. ["1.0", "0.0", "0.0"])
            default_val = value if value else "0.0"
            self.values = [default_val] * self.num_text_inputs
        else:
            self.values = []

        self.active_input_index = -1 # Welches der Textfelder gerade editiert wird (-1 = keins)
        self.cursor_pos = 0

        self.x = x
        self.y = y
        self.is_dragging = False

        self.is_nesting = is_nesting
        self.num_slots = num_slots if is_nesting else 0
        self.slots = [None] * self.num_slots
        self.parent_nest_block = None
        self.parent_slot_index = -1
        
        # Pointers for top-to-bottom stacking
        self.prev_block = None
        self.next_block = None

        # Cache containers for GPU batches
        self.batch_body = None
        self.batch_outline = None
        self.last_w = -1
        self.last_h = -1

        self.update_layout()

    def update_gpu_batches(self):
        if self.width == self.last_w and self.height == self.last_h and self.batch_body is not None:
            return

        shader = get_block_shader()

        # main body batch 
        vertices = ((0, 0), (self.width, 0), (0, -self.height), (self.width, -self.height))
        indices = ((0, 1, 2), (2, 1, 3))
        self.batch_body = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices = indices)

        # outline batch
        outline_verts = ((0, 0), (self.width, 0), (self.width, -self.height), (0, -self.height))
        outline_indices = ((0, 1), (1, 2), (2, 3), (3, 0))
        self.batch_outline = batch_for_shader(shader, 'LINES', {"pos": outline_verts}, indices = outline_indices)

        self.last_w = self.width
        self.last_h = self.height
        

    def get_input_rects(self):
        """Gibt eine Liste von Rechtecken für alle Text-Inputs zurück."""
        if self.num_text_inputs <= 0:
            return []
        rects = []
        label_w = 85  # Platz für das Label
        start_x = self.x + label_w
        total_w = self.width - label_w - 10
        box_w = (total_w - (self.num_text_inputs - 1) * 4) / self.num_text_inputs
        box_h = 24
        box_y = self.y - 8
        for i in range(self.num_text_inputs):
            bx = start_x + i * (box_w + 4)
            rects.append((bx, box_y, box_w, box_h))
        return rects

    def get_slot_rects(self):
        """Returns [(slot_x, slot_y, slot_w, slot_h), ...] for each slot."""
        if not self.is_nesting or not self.slots:
            return []

        rects = []
        cur_x = self.x + 55

        for child in self.slots:
            if child:
                slot_w = child.width
                slot_h = child.height
            else:
                slot_w = 46
                slot_h = 24 # empty slot placeholder

            # center slot vertically
            slot_y = self.y - ((self.height - slot_h) / 2)
            rects.append((cur_x, slot_y, slot_w, slot_h))
            cur_x += slot_w + 8

        return rects

    def update_layout(self, update_parent=True):
        if self.is_nesting:
            total_slots_w = 0
            max_slot_h = 24
            for child in self.slots:
                if child:
                    child.update_layout(update_parent=False)
                    total_slots_w += child.width
                    max_slot_h = max(max_slot_h, child.height)
                else:
                    total_slots_w += 46
            gaps = max(0, len(self.slots) - 1) * 8
            self.width = 55 + total_slots_w + gaps + 12
            self.height = max(42, max_slot_h + 16)
            slot_rects = self.get_slot_rects()
            for i, child in enumerate(self.slots):
                if child and i < len(slot_rects):
                    sx, sy, _, _ = slot_rects[i]
                    child.x = sx
                    child.y = sy
                    child.update_position(0, 0)
        elif self.num_text_inputs > 0:
            self.width = max(240, 90 + self.num_text_inputs * 55)
            self.height = 48
        else:
            self.width = 160
            self.height = 34

        # whenever layout updates, refresh GPU batches
        self.update_gpu_batches()

        if update_parent and self.parent_nest_block:
            self.parent_nest_block.update_layout()




    def get_clicked_input_index(self, mx, my):
        rects = self.get_input_rects()
        for i, (bx, by, bw, bh) in enumerate(rects):
            if bx <= mx <= bx + bw and (by - bh) <= my <= by:
                return i
        return -1

    def unsnap(self):
        """Detaches this block from anything above it."""
        if self.prev_block:
            self.prev_block.next_block = None
            self.prev_block = None

        if self.parent_nest_block:
            parent = self.parent_nest_block
            slot_idx = self.parent_slot_index
            if 0 <= slot_idx < len(parent.slots) and parent.slots[slot_idx] == self:
                parent.slots[slot_idx] = None
            self.parent_nest_block = None
            self.parent_slot_index = -1
            parent.update_layout()
            parent.update_position(0, 0)

    def update_position(self, dx, dy):
        """Recursively moves this block, its nested slot blocks, and stacked blocks."""
        self.x += dx
        self.y += dy
        if self.is_nesting:
            slot_rects = self.get_slot_rects()
            for i, child in enumerate(self.slots):
                if child and i < len(slot_rects):
                    sx, sy, _, _ = slot_rects[i]
                    child_dx = sx - child.x
                    child_dy = sy - child.y
                    child.update_position(child_dx, child_dy)
        if self.next_block:
            target_x = self.x
            target_y = self.y - self.height - 4
            next_dx = target_x - self.next_block.x
            next_dy = target_y - self.next_block.y
            self.next_block.update_position(next_dx, next_dy)


# Global list storing all blocks on our canvas
canvas_blocks = []

#def draw_rect(x, y, w, h, color):
#    shader = blockShader #gpu.shader.from_builtin('UNIFORM_COLOR')
#    vertices = ((x, y), (x + w, y), (x, y - h), (x + w, y - h))
#    indices = ((0, 1, 2), (2, 1, 3))
#    batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices = #indices)
#    shader.bind()
#    shader.uniform_float("color", color)
#    batch.draw(shader)
#
#def draw_rect_outline(x, y, w, h, color=(0.0, 0.0, 0.0, 1.0), thickness=2.0):
#    shader = blockShader #gpu.shader.from_builtin('UNIFORM_COLOR')
#    vertices = ((x, y), (x + w, y), (x + w, y - h), (x, y - h))
#    indices = ((0, 1), (1, 2), (2, 3), (3, 0))
#    batch = batch_for_shader(shader, 'LINES', {"pos": vertices}, #indices=indices)
#    gpu.state.line_width_set(thickness)
#    shader.bind()
#    shader.uniform_float("color", color)
#    batch.draw(shader)
#    gpu.state.line_width_set(1.0)

def draw_cashed_rect(x, y, batch, color, thickness=None):
    if not batch:
        return
    shader = get_block_shader()
    shader.bind()
    shader.uniform_float("color", color)
    
    # Optional line thickness handling for outlines
    if thickness is not None:
        gpu.state.line_width_set(thickness)
        
    # Use push/pop matrix stack
    gpu.matrix.push()
    gpu.matrix.translate((x, y))
    batch.draw(shader)
    gpu.matrix.pop()

    if thickness is not None:
        gpu.state.line_width_set(1.0)

def draw_rect_optimized(x, y, w, h, color):
    shader = get_block_shader()
    vertices = ((x, y), (x + w, y), (x, y - h), (x + w, y - h))
    indices = ((0, 1, 2), (2, 1, 3))
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)

def draw_rect_outline_optimized(x, y, w, h, color=(0.0, 0.0, 0.0, 1.0), thickness=2.0):
    shader = get_block_shader()
    vertices = ((x, y), (x + w, y), (x + w, y - h), (x, y - h))
    indices = ((0, 1), (1, 2), (2, 3), (3, 0))
    batch = batch_for_shader(shader, 'LINES', {"pos": vertices}, indices=indices)
    gpu.state.line_width_set(thickness)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.line_width_set(1.0)

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

    gpu.state.blend_set('NONE')

# ---   EVALUATION ENGINE ---

def evaluate_slot_value(slot_block, context_labels):
    if not slot_block:
        return [0.0, 0.0, 0.0]

    if slot_block.block_type in {"VECTOR_MATH", "VECTOR"}:
        try:
            return [float(slot_block.values[0]), 
                    float(slot_block.values[1]), 
                    float(slot_block.values[2])]
        except ValueError:
            return [0.0, 0.0, 0.0]

    elif slot_block.block_type == "GET_LABEL":
        var_name = slot_block.values[0] if slot_block.values else ""
        val = context_labels.get(var_name, [0.0, 0.0, 0.0])
        return val if isinstance(val, list) else [0.0, 0.0, 0.0]

    return [0.0, 0.0, 0.0]

def evaluate_block_stack(stack_blocks, context_labels):
    current_data = None
    for curr in stack_blocks:
        if curr.block_type == "INPUT_GEO":
            obj_name = curr.values[0] if curr.values and curr.values[0] else ""
            if obj_name and obj_name in bpy.data.objects:
                current_data = bpy.context.active_object
        elif curr.block_type == "MESH_CUBE":
            current_data = {"translation": [0.0, 0.0, 0.0]}
        elif curr.block_type in {"VECTOR_MATH", "VECTOR"}:
            try:
                current_data = [float(curr.values[0]), float(curr.values[1]), float(curr.values[2])]
            except ValueError:
                current_data = [0.0, 0.0, 0.0]
        elif curr.block_type == "SET_LABEL":
            var_name = curr.values[0] if curr.values else "var"
            context_labels[var_name] = current_data
        elif curr.block_type == "GET_LABEL":
            var_name = curr.values[0] if curr.values else "var"
            current_data = context_labels.get(var_name, current_data)
        elif curr.block_type == "TRANSFORM":
            translation = [0.0, 0.0, 0.0]
            if curr.is_nesting and len(curr.slots) > 0 and curr.slots[0] is not None:
                translation = evaluate_slot_value(curr.slots[0], context_labels)

                # apply transformation to target object
                target_obj = current_data if isinstance(current_data, bpy.types.Object) else bpy.context.active_object
                if target_obj:
                    target_obj.location = [target_obj.location[i] + translation[i] for i in range(3)]
                    current_data = target_obj

        elif curr.block_type == "GEO_OUTPUT":
            pass
    return current_data

def run_evaluation_pipeline():
    context_labels = {}

    if not canvas_blocks:
        print("ScratchBlocks: No blocks present")
        return

    # 1. Evaluate all independent stacks ending in SET_LABEL first so variables are populated
    set_label_blocks = [b for b in canvas_blocks if b.block_type == "SET_LABEL"]
    for set_block in set_label_blocks:
        sub_stack = []
        curr = set_block
        while curr:
            sub_stack.insert(0, curr)
            curr = curr.prev_block
        evaluate_block_stack(sub_stack, context_labels)

    # 2. Evaluate the main output stack ending in GEO_OUTPUT
    output_block = next((b for b in canvas_blocks if b.block_type == "GEO_OUTPUT"), None)
    if not output_block:
        print("ScratchBlocks: No output block present")
        return

    stack_blocks = []
    curr = output_block

    while curr:
        stack_blocks.insert(0, curr)
        curr = curr.prev_block

    print(f"--- SCRATCH BLOCKS EVALUATION --- Gefundene Kette: {[b.block_type for b in stack_blocks]}")

    final_data = evaluate_block_stack(stack_blocks, context_labels)

    if final_data:
        print(f"Scratch Graph evaluated successfully on object: {getattr(final_data, 'name', 'None')}")
            
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
            new_block = ScratchBlockData("NEW_BLOCK", x = event.mouse_region_x - 80, y = event.mouse_region_y + 20)
            canvas_blocks.append(new_block)
            context.area.tag_redraw()
            return {'FINISHED'}
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
        op.num_text_inputs = 1
        op.value = ""
        op.is_nesting = False
        op.color = (0.20, 0.50, 0.70, 1.0)


        # 2. Mesh Primitive (Cube) - Source (Grün)
        op = layout.operator("scratch.add_block", text="Mesh Primitive (Cube)", icon='MESH_CUBE')
        op.block_type = "MESH_CUBE"
        op.label = "Mesh (Cube)"
        op.num_text_inputs = 0
        op.is_nesting = False
        op.is_set_label = False
        op.is_get_label = False
        op.color = (0.22, 0.60, 0.35, 1.0) 

        # 2. Vector Math / Vector Input (Blau, mit 3 Float-Inputs für X, Y, Z)
        op = layout.operator("scratch.add_block", text="Vector Math", icon='CON_LOCLIKE')
        op.block_type = "VECTOR_MATH"
        op.label = "Vector"
        op.num_text_inputs = 3
        op.value = "0.0"
        op.is_nesting = False
        op.is_set_label = False
        op.is_get_label = False
        op.color = (0.25, 0.55, 0.85, 1.0) 

        # 3. Transform Geometry - Modifier (Orange, 1 Slot für Translation)
        op = layout.operator("scratch.add_block", text="Transform Geometry", icon='OBJECT_DATA')
        op.block_type = "TRANSFORM"
        op.label = "Transform"
        op.num_text_inputs = 0
        op.is_nesting = True
        op.num_slots = 1
        op.is_set_label = False
        op.is_get_label = False
        op.color = (0.90, 0.60, 0.10, 1.0) 

        # 4. Group Output - Sink (Rot)
        op = layout.operator("scratch.add_block", text="Group Output", icon='EXPORT')
        op.block_type = "GEO_OUTPUT"
        op.label = "Group Output"
        op.num_text_inputs = 0
        op.is_nesting = False
        op.is_set_label = False
        op.is_get_label = False
        op.color = (0.85, 0.25, 0.25, 1.0) 

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
        op.color = (0.85, 0.35, 0.20, 1.0)

        # 6. Get Label Block (Hellblau)
        op = layout.operator("scratch.add_block", text="Get Label", icon='EYEDROPPER')
        op.block_type = "GET_LABEL"
        op.label = "Get:"
        op.num_text_inputs = 1
        op.is_nesting = False
        op.is_set_label = False
        op.is_get_label = True
        op.value = "my_geo"
        op.color = (0.20, 0.50, 0.80, 1.0)
    
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