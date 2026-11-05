import bpy
import math
import json
from .scratch_data import canvas_blocks, get_current_loaded_graph_name, load_graph, save_current_graph, ScratchBlockData

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
    adjacency_list = {i: [] for i in range(n)}

    # adjacency_list[0] = [1, 3]: Stack 1 and Stack 3 depends on stack 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # If stack j reads any variable written by stack i, then i must run before j
            if stack_meta[i]['writes'] & stack_meta[j]['reads']: # '&': set intersection
                # 'writes' and 'reads' are lists
                adjacency_list[i].append(j)
                in_degree[j] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    sorted_indices = []

    while queue:
        curr_idx = queue.pop(0)
        sorted_indices.append(curr_idx)

        for neighbor in adjacency_list[curr_idx]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_indices) != n:
        print("Warning: Circular depandency detected in Scratch blocks. Falling back to default order.")
        return stacks

    return [stacks[i] for i in sorted_indices]



def evaluate_block_stack(stack_blocks, context_labels, owner_obj=None):

    current_data = None
    depsgraph = bpy.context.evaluated_depsgraph_get()

    for curr in stack_blocks:
        if curr.block_type == "INPUT_GEO":
            target_obj = owner_obj if owner_obj else bpy.context.active_object
            
            if target_obj and target_obj.type == 'MESH':
                obj_evaluated = target_obj.evaluated_get(depsgraph)
                eval_mesh = obj_evaluated.to_mesh()
                # CRITICAL: Clean up the temporary mesh from memory when done
                obj_evaluated.to_mesh_clear()
                current_data = target_obj

        elif curr.block_type in {"MESH_CUBE", "MESH_CYLINDER"}:
            # 1. Determine target object to reuse or replace
            target_obj = owner_obj if owner_obj else bpy.context.active_object
            if target_obj and target_obj.type == 'MESH':
                if curr.block_type == "MESH_CYLINDER":
                    bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, location=(0, 0, 0))
                else:
                    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))

                temp_obj = bpy.context.active_object

                old_mesh = target_obj.data
                target_obj.data = temp_obj.data
                bpy.data.objects.remove(temp_obj, do_unlink=True)

                if old_mesh.users == 0:
                    bpy.data.meshes.remove(old_mesh)

                current_data = target_obj
                curr.target_object_name = target_obj.name
            else:
                current_data = None
                       


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
                    target_obj["scratch_base_loc"] = list(target_obj.location)

                # apply transformation
                target_obj.location = (
                    target_obj["scratch_base_loc"][0] + translation[0],
                    target_obj["scratch_base_loc"][1] + translation[1], 
                    target_obj["scratch_base_loc"][2] + translation[2]
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
    # Save the current active canvas state first
    current_graph = get_current_loaded_graph_name()
    active_obj = bpy.context.active_object
    if current_graph and active_obj and getattr(active_obj, "scratch_graph_name", "") == current_graph:
        save_current_graph(current_graph)
    
    # Find all objects in the scene that have a scratch graph assigned
    objects_with_graphs = [obj for obj in bpy.data.objects if obj.get("scratch_graph_name")]

    if not objects_with_graphs:
        print("ScratchBlocks: No objects with graphs found.")
        reset_unlinked_scratch_objects(set())
        return

    # Initialise base state
    for obj in bpy.data.objects:
        if obj.get("scratch_graph_name") or "scratch_base_initialized" in obj:
            if "scratch_base_initialized" not in obj:
                obj["scratch_base_loc"] = list(obj.location)
                obj["scratch_base_rot"] = list(obj.rotation_euler)
                obj["scratch_base_scale"] = list(obj.scale)
                if obj.type == 'MESH' and obj.data:
                    base_mesh = obj.data.copy()
                    base_mesh.use_fake_user = True
                    obj["scratch_base_mesh_name"] = base_mesh.name
                obj["scratch_base_initialized"] = True

    # Reset all tracked objects to their base location before evaluation loop
    for obj in bpy.data.objects:
        if obj.get("scratch_base_initialized"):
            if "scratch_base_loc" in obj:
                base_loc = obj["scratch_base_loc"]
                obj.location = (base_loc[0], base_loc[1], base_loc[2])
            if "scratch_base_rot" in obj:
                rot = obj["scratch_base_rot"]
                obj.rotation_euler = (rot[0], rot[1], rot[2])
            if "scratch_base_scale" in obj:
                scale = obj["scratch_base_scale"]
                obj.scale = (scale[0], scale[1], scale[2])

            mesh_name = obj.get("scratch_base_mesh_name")
            if mesh_name and mesh_name in bpy.data.meshes:
                base_mesh = bpy.data.meshes[mesh_name]
                if obj.type == 'MESH' and obj.data != base_mesh:
                    old_mesh = obj.data
                    obj.data = base_mesh
                    if old_mesh and old_mesh.users == 0 and old_mesh.name != mesh_name:
                        bpy.data.meshes.remove(old_mesh)

    evaluated_objects = set()
    final_data = None

    # Evaluate each object's graph sequentially
    for obj in objects_with_graphs:
        graph_name = obj.get("scratch_graph_name")

        # 1. If this graph is currently open on the canvas, use live canvas blocks
        if graph_name == current_graph:
            if not canvas_blocks:
                continue
            eval_blocks = canvas_blocks
        else:
            # 2. For other objects, load temporarily from their text data-block
            text_block = bpy.data.texts.get(graph_name)
            if not text_block:
                print(f"ERROR: text datablock '{graph_name}' not found!")
                print("Available text datablicks:")
                for t in bpy.data.texts:
                    print(f"   {t.name}")
                continue

            # Temporarily parse/load blocks for this object graph without disrupting UI
            try:
                graph_data = json.loads(text_block.as_string())
                print(f"Successfully loaded {len(graph_data)} blocks.")
            except Exception as e:
                print(f"ERROR parsing graph '{graph_name}': {e}")
                continue

            eval_blocks = []
            for data in graph_data:
                b = ScratchBlockData(
                    block_type = data["block_type"],
                    x = data["x"], y = data["y"],
                    label = data["label"], 
                    color = tuple(data["color"]),
                    value = data["values"][0] if data["values"] else "",
                    num_text_inputs = len(data["values"]),
                    is_nesting = data["is_nesting"],
                    num_slots = data.get("num_slots", 2),
                    has_display_text = data.get("has_display_text", False),
                    target_object_name = data.get("target_object_name")
                )
                b.values = data["values"]
                eval_blocks.append(b)

            # Restore block links for background evaluation
            for i, data in enumerate(graph_data):
                block = eval_blocks[i]
                next_idx = data.get("next_block_idx", -1)
                if 0 <= next_idx < len(eval_blocks):
                    block.next_block = eval_blocks[next_idx]
                    eval_blocks[next_idx].prev_block = block

                slot_indices = data.get("slot_indices", [])
                if block.is_nesting and slot_indices:
                    for slot_idx, child_idx in enumerate(slot_indices):
                        if 0 <= child_idx < len(eval_blocks):
                            child_block = eval_blocks[child_idx]
                            block.slots[slot_idx] = child_block
                            child_block.parent_nest_block = block
                            child_block.parent_slot_index = slot_idx

            for b in eval_blocks:
                if b.parent_nest_block is None:
                    b.update_layout(update_parent = False)

        all_stacks = get_all_stacks(eval_blocks)
        active_stacks = get_active_stacks(all_stacks)
        ordered_stacks = sort_stacks_topologically(active_stacks)

        # print(f"Object: {obj.name}")
        # print(f"Graph: {graph_name}")
        # print(f"Blocks loaded: {len(eval_blocks)}")
        # print(f"All stacks: {len(all_stacks)}")
        # print(f"Active stacks: {len(active_stacks)}")
        # print(f"Ordered stacks: {len(ordered_stacks)}")

        context_labels = {}
        final_data = None
        for stack in ordered_stacks:
            result = evaluate_block_stack(stack, context_labels, owner_obj=obj)
            if any(b.block_type == "GEO_OUTPUT" for b in stack):
                final_data = result

        if isinstance(final_data, bpy.types.Object):
            evaluated_objects.add(final_data)

    # Clean up unlinked objects only after ALL graphs have finished evaluating
    reset_unlinked_scratch_objects(evaluated_objects)


def reset_unlinked_scratch_objects(evaluated_objects):
    for obj in bpy.data.objects:
        if obj.get("scratch_base_initialized"):
            if obj not in evaluated_objects:
                if "scratch_base_loc" in obj:
                    loc = obj["scratch_base_loc"]
                    obj.location = (loc[0], loc[1], loc[2])
                if "scratch_base_rot" in obj:
                    rot = obj["scratch_base_rot"]
                    obj.rotation_euler = (rot[0], rot[1], rot[2])
                if "scratch_base_scale" in obj:
                    scale = obj["scratch_base_scale"]
                    obj.scale = (scale[0], scale[1], scale[2])

                mesh_name = obj.get("scratch_base_mesh_name")
                if mesh_name and mesh_name in bpy.data.meshes:
                    base_mesh = bpy.data.meshes[mesh_name]
                    if obj.type == 'MESH' and obj.data != base_mesh:
                        old_mesh = obj.data
                        obj.data = base_mesh
                        if old_mesh and old_mesh.users == 0 and old_mesh.name != mesh_name:
                            bpy.data.meshes.remove(old_mesh)


        if "scratch_base_loc" in obj:
            if obj not in evaluated_objects:
                base_loc = obj["scratch_base_loc"]
                obj.location = (base_loc[0], base_loc[1], base_loc[2])
                del obj["scratch_base_loc"]