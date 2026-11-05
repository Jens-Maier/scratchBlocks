import bpy
import gpu
from gpu_extras.batch import batch_for_shader
#from .scratch_logic import run_evaluation_pipeline
import json

canvas_blocks = []
last_mouse_pos = [100, 200]
_is_modal_running = False
cachedShader = None

def get_canvas_blocks():
    return canvas_blocks

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
    def __init__(self, block_type, x=50, y=100, label=None, color=None, has_text_input=False, value="", num_text_inputs=0, is_nesting=False, num_slots=2, is_set_label=False, is_get_label=False, has_display_text=False, target_object_name = None):

        self.block_type = block_type
        self.label = label or block_type
        self.color = color if color is not None else (0.9, 0.6, 0.1, 1.0)
        self.target_object_name = target_object_name
        self.has_display_text = has_display_text

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


_current_loaded_graph_name = None

def get_current_loaded_graph_name():
    return _current_loaded_graph_name

def set_current_loaded_graph_name(name):
    global _current_loaded_graph_name
    _current_loaded_graph_name = name

def save_current_graph(graph_name):
    if not graph_name:
        return
    
    # Map each block instance to its index in canvas_blocks
    block_to_idx = {b: i for i, b in enumerate(canvas_blocks)}

    # Serialize all blocks on the canvas into a JSON-compatible structure
    graph_data = []
    for b in canvas_blocks:
        next_idx = block_to_idx.get(b.next_block, -1) if hasattr(b, 'next_block') else -1

        slot_indices = []
        if getattr(b, 'is_nesting', False):
            for child in b.slots:
                slot_indices.append(block_to_idx.get(child, -1) if child else -1)

        data = {
            "block_type": b.block_type,
            "label": b.label,
            "x": b.x,
            "y": b.y,
            "values": b.values,
            "is_nesting": b.is_nesting,
            "color": list(b.color),
            "has_display_text": getattr(b, 'has_display_text', False),
            "target_object_name": getattr(b, 'target_object_name', None),
            "next_block_idx": next_idx,
            "slot_indices": slot_indices,
            "num_slots": len(b.slots) if b.is_nesting else 0,
        }
        graph_data.append(data)
    
    json_str = json.dumps(graph_data, indent=2)
    
    # Save into a Blender Text data-block (serves as our graph file/asset)
    text_block = bpy.data.texts.get(graph_name)
    if not text_block:
        text_block = bpy.data.texts.new(name=graph_name)
    text_block.clear()
    text_block.write(json_str)
    
    
def load_graph(graph_name):
    global canvas_blocks, _current_loaded_graph_name

    # Save current graph before switching away
    if _current_loaded_graph_name and _current_loaded_graph_name != graph_name:
        save_current_graph(_current_loaded_graph_name)

    _current_loaded_graph_name = graph_name
    text_block = bpy.data.texts.get(graph_name)
    if not text_block:
        canvas_blocks.clear()
        return

    try:
        graph_data = json.loads(text_block.as_string())
    except json.JSONDecodeError:
        print("Failed to parse graph JSON.")
        return

    # Clear current canvas
    canvas_blocks.clear()

    # Rebuild blocks from saved data
    for data in graph_data:
        new_block = ScratchBlockData(
            block_type=data["block_type"],
            x=data["x"],
            y=data["y"],
            label=data["label"],
            color=tuple(data["color"]),
            value=data["values"][0] if data["values"] else "",
            num_text_inputs=len(data["values"]),
            is_nesting=data["is_nesting"], 
            num_slots = data.get("num_slots", 2),
            has_display_text=data.get("has_display_text", False),
            target_object_name=data.get("target_object_name")
        )
        new_block.values = data["values"]
        canvas_blocks.append(new_block)

    # Restore connections
    for i, data in enumerate(graph_data):
        block = canvas_blocks[i]
        next_idx = data.get("next_block_idx", -1)
        if 0 <= next_idx <len(canvas_blocks):
            block.next_block = canvas_blocks[next_idx]
            canvas_blocks[next_idx].prev_block = block

        slot_indices = data.get("slot_indices", [])
        if block.is_nesting and slot_indices:
            for slot_idx, child_idx in enumerate(slot_indices):
                if 0 <= child_idx < len(canvas_blocks):
                    child_block = canvas_blocks[child_idx]
                    block.slots[slot_idx] = child_block
                    child_block.parent_nest_block = block
                    child_block.parent_slot_index = slot_idx

    for b in canvas_blocks:
        if b.parent_nest_block is None:
            b.update_layout(update_parent = False)

    # Force area redraw
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()
                
    print(f"Graph '{graph_name}' loaded successfully.")


def update_scratch_graph_property(self, context):
    if self == context.active_object and self.scratch_graph_name:
        load_graph(self.scratch_graph_name)