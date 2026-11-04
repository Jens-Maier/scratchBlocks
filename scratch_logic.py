import bpy
import math
from .scratch_data import canvas_blocks

def to_vec3(val):
    if isinstance(val, (int, float)):
        return [float(val), float(val), float(val)]
    elif isinstance(val, list):
        if len(val) >= 3:
            return [float(val[0]), float(val[1]), float(val[2])]
        elif len(val) == 2:
            return [float(val[0]), float(val[1]), 0.0]
        elif len(val) == 1:
            return [float(val[0]), float(val[0]), float(val[0])]
    return [0.0, 0.0, 0.0]



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

    elif slot_block.block_type == "MATH_ADD":
        v1 = to_vec3(evaluate_slot_value(slot_block.slots[0], context_labels) if len(slot_block.slots) > 0 else 0.0)
        v2 = to_vec3(evaluate_slot_value(slot_block.slots[1], context_labels) if len(slot_block.slots) > 1 else 0.0)
        return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]

    elif slot_block.block_type == "MATH_MULTIPLY":
        v1 = to_vec3(evaluate_slot_value(slot_block.slots[0], context_labels) if len(slot_block.slots) > 0 else 0.0)
        v2 = to_vec3(evaluate_slot_value(slot_block.slots[1], context_labels) if len(slot_block.slots) > 1 else 0.0)
        return [v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]]

    elif slot_block.block_type == "MATH_SIN":
        v1 = to_vec3(evaluate_slot_value(slot_block.slots[0], context_labels) if len(slot_block.slots) > 0 else 0.0)
        return [math.sin(v1[0]), math.sin(v1[1]), math.sin(v1[2])]
    
    return [0.0, 0.0, 0.0]


def get_all_stacks(canvas_blocks):
    """Finds all distinct terminal blocks (SET_LABEL or GEO_OUTPUT) and traces back their full stacks."""
    all_prevs = {b.prev_block for b in canvas_blocks if hasattr(b, 'prev_block') and b.prev_block is not None}

    terminal_blocks = [
        b for b in canvas_blocks 
        if b.block_type in {"SET_LABEL", "GEO_OUTPUT"} and b not in all_prevs
    ]

    stacks = []
    for term in terminal_blocks:
        stack = []
        curr = term
        while curr:
            stack.insert(0, curr)
            curr = curr.prev_block
        stacks.append(stack)
    return stacks

def get_active_stacks(all_stacks):
    """Filters all stacks to only those required by GEO_OUTPUT chains via variable dependencies."""
    output_stacks = [s for s in all_stacks if any(b.block_type == "GEO_OUTPUT" for b in s)]
    if not output_stacks:
        return []

    # Map each variable name to the stack(s) that write it
    var_writer_map = {}
    for stack in all_stacks:
        writes, _ = analyze_stack_dependencies(stack)
        for w in writes:
            if w not in var_writer_map:
                var_writer_map[w] = []
            var_writer_map[w].append(stack)

    active_stacks = set()
    queue = list(output_stacks)
    visited = set()

    while queue:
        curr_stack = queue.pop(0)
        stack_key = tuple(curr_stack)
        if stack_key in visited:
            continue
        visited.add(stack_key)
        active_stacks.add(stack_key)

        # Check what variables this stack reads, and queue their writer stacks
        _, reads = analyze_stack_dependencies(curr_stack)
        for r in reads:
            if r in var_writer_map:
                for writer_stack in var_writer_map[r]:
                    if tuple(writer_stack) not in visited:
                        queue.append(writer_stack)

    # Preserve original relative order for active stacks
    return [s for s in all_stacks if tuple(s) in active_stacks]

def analyze_stack_dependencies(stack_blocks):
    writes = set()
    reads = set()

    def scan_slot(slot_block):
        if not slot_block:
            return
        if slot_block.block_type == "GET_LABEL":
            var_name = slot_block.values[0] if slot_block.values else ""
            if var_name:
                reads.add(var_name)
        for s in getattr(slot_block, 'slots', []):
            scan_slot(s)

    for curr in stack_blocks:
        if curr.block_type == "SET_LABEL":
            var_name = curr.values[0] if curr.values else "var"
            writes.add(var_name)
        elif curr.block_type == "GET_LABEL":
            var_name = curr.values[0] if curr.values else "var"
            if var_name:
                reads.add(var_name)

        if hasattr(curr, 'slots'):
            for s in curr.slots:
                scan_slot(s)

    return writes, reads

def sort_stacks_topologically(stacks):
    stack_meta = []
    for idx, stack in enumerate(stacks):
        writes, reads = analyze_stack_dependencies(stack)
        stack_meta.append({
            'index': idx,
            'stack': stack,
            'writes': writes,
            'reads': reads
        })

    n = len(stacks)
    in_degree = {i: 0 for i in range(n)} # {0: 0, 1: 0, 2: 0} (dictionary)
    adj = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # If stack j reads any variable written by stack i, then i must run before j
            if stack_meta[i]['writes'] & stack_meta[j]['reads']:
                adj[i].append(j)
                in_degree[j] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    sorted_indices = []

    while queue:
        curr_idx = queue.pop(0)
        sorted_indices.append(curr_idx)

        for neighbor in adj[curr_idx]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_indices) != n:
        print("Warning: Curcular depandency detected in Scratch blocks. Falling back to default order.")
        return stacks

    return [stacks[i] for i in sorted_indices]



def evaluate_block_stack(stack_blocks, context_labels):

    current_data = None
    depsgraph = bpy.context.evaluated_depsgraph_get()

    for curr in stack_blocks:
        if curr.block_type == "INPUT_GEO":
            obj_name = curr.values[0] if curr.values and curr.values[0] else ""
            if obj_name and obj_name in bpy.data.objects:
                target_obj = bpy.data.objects[obj_name]
            else:
                target_obj = bpy.context.active_object

            if target_obj and target_obj.type == 'MESH':
                obj_evaluated = target_obj.evaluated_get(depsgraph)

                eval_mesh = obj_evaluated.to_mesh()

                # CRITICAL: Clean up the temporary mesh from memory when done
                obj_evaluated.to_mesh_clear()

                current_data = target_obj

        elif curr.block_type == "MESH_CUBE":
            bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
            current_data = bpy.context.active_object

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

            if isinstance(current_data, bpy.types.Object):    
                # apply transformation to target object # -> s. gemini...
                target_obj = current_data

                if "scratch_base_loc" not in target_obj:
                    target_obj["scratch_base_loc"] = [target_obj.location.x, target_obj.location.y, target_obj.location.z]


                # apply transformation
                target_obj.location = (
                    target_obj.location[0] + translation[0],
                    target_obj.location[1] + translation[1], 
                    target_obj.location[2] + translation[2]
                )

                obj_evaluated = target_obj.evaluated_get(depsgraph)
                eval_mesh = obj_evaluated.to_mesh()
                obj_evaluated.to_mesh_clear()

                current_data = target_obj
            else:
                # no input object connected
                current_data = None

    
        elif curr.block_type == "GEO_OUTPUT":
            pass
    return current_data





def run_evaluation_pipeline():
    # reset all tracked objects to their base location
    for obj in bpy.data.objects:
        if "scratch_base_loc" in obj:
            base_loc = obj["scratch_base_loc"]
            obj.location = (base_loc[0], base_loc[1], base_loc[2])
            del obj["scratch_base_loc"]

    context_labels = {}

    if not canvas_blocks:
        print("ScratchBlocks: No blocks present")
        reset_unlinked_scratch_objects(active_evaluated_obj=None)
        return

    # 1. Extract all independent/chained statement stacks
    all_stacks = get_all_stacks(canvas_blocks)

    # 2. Filter out unreferenced branches (dead code elimination)
    active_stacks = get_active_stacks(all_stacks)

    # 3. Sort stacks topologically based on variable dependencies
    ordered_stacks = sort_stacks_topologically(active_stacks)

    final_data = None

    # 4. Evaluate stacks in correct sequence
    for stack in ordered_stacks:
        print(f"--- STACK EVALUATION --- Chain: {[b.block_type for b in stack]}")
        result = evaluate_block_stack(stack, context_labels)
        if any(b.block_type == "GEO_OUTPUT" for b in stack):
            final_data = result

    reset_unlinked_scratch_objects(active_evaluated_obj=final_data)

    if final_data:
        print(f"Scratch Graph evaluated successfully on object: {getattr(final_data, 'name', 'None')}")
    
    


def reset_unlinked_scratch_objects(active_evaluated_obj):
    for obj in bpy.data.objects:
        if "scratch_base_loc" in obj:
            if obj != active_evaluated_obj:
                base_loc = obj["scratch_base_loc"]
                obj.location = (base_loc[0], base_loc[1], base_loc[2])
                del obj["scratch_base_loc"]