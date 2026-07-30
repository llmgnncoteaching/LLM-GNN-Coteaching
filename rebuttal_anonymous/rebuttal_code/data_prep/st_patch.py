import re
f = "/project/anon/rebuttal/port/LG-CoTeaching/create_co_teaching_data.py"
s = open(f).read()
if "self_training" in s:
    print("already patched")
    raise SystemExit
s = s.replace(
    '    parser.add_argument("--dataset", type=str, required=True)',
    '    parser.add_argument("--dataset", type=str, required=True)\n    parser.add_argument("--self_training", type=int, default=0, help="1=self-training ablation")',
    1)
old = (
'    write_sft_data(\n'
'        gnn_for_llm, graph_data, args.dataset, args.sft_output_path,\n'
'        include_train_nodes=True, anchor_repeat=args.anchor_repeat,\n'
'        use_neighbor_info=bool(args.use_neighbor_info), adj=adj, adj2=adj2,\n'
'    )\n'
'\n'
'    # === Cross-selection 2: LLM selects clean labels FOR GNN ===\n'
'    print(f"\\n--- LLM cross-selects for GNN (top {select_ratio:.0%}) ---")\n'
'    llm_for_gnn, llm_stats = cross_select_llm_for_gnn(\n'
'        gnn_predictions, llm_predictions, covered_ids,\n'
'        graph_data, select_ratio, llm_logprobs=llm_logprobs,\n'
'    )\n'
'    print(f"  Selected: {llm_stats[\'n_selected\']}/{llm_stats[\'total_candidates\']} "\n'
'          f"(agreed: {llm_stats[\'n_agreed_in_selected\']}, "\n'
'          f"acc: {llm_stats[\'pseudo_label_accuracy\']:.4f}, "\n'
'          f"avg_logprob: {llm_stats.get(\'avg_llm_logprob\', \'N/A\')})")\n'
'\n'
'    write_gnn_pseudo_labels(llm_for_gnn, graph_data, args.gnn_pseudo_label_path)'
)
new = (
'    # === Cross-selection 2: LLM selects clean labels FOR GNN ===\n'
'    print(f"\\n--- LLM cross-selects for GNN (top {select_ratio:.0%}) ---")\n'
'    llm_for_gnn, llm_stats = cross_select_llm_for_gnn(\n'
'        gnn_predictions, llm_predictions, covered_ids,\n'
'        graph_data, select_ratio, llm_logprobs=llm_logprobs,\n'
'    )\n'
'    print(f"  Selected: {llm_stats[\'n_selected\']}/{llm_stats[\'total_candidates\']} "\n'
'          f"(agreed: {llm_stats[\'n_agreed_in_selected\']}, "\n'
'          f"acc: {llm_stats[\'pseudo_label_accuracy\']:.4f}, "\n'
'          f"avg_logprob: {llm_stats.get(\'avg_llm_logprob\', \'N/A\')})")\n'
'\n'
'    if args.self_training:\n'
'        sft_src, pseudo_src = llm_for_gnn, gnn_for_llm\n'
'        print("  [self_training] each model self-labels")\n'
'    else:\n'
'        sft_src, pseudo_src = gnn_for_llm, llm_for_gnn\n'
'\n'
'    write_sft_data(\n'
'        sft_src, graph_data, args.dataset, args.sft_output_path,\n'
'        include_train_nodes=True, anchor_repeat=args.anchor_repeat,\n'
'        use_neighbor_info=bool(args.use_neighbor_info), adj=adj, adj2=adj2,\n'
'    )\n'
'    write_gnn_pseudo_labels(pseudo_src, graph_data, args.gnn_pseudo_label_path)'
)
if old not in s:
    print("OLD_NOT_FOUND")
    raise SystemExit
s = s.replace(old, new, 1)
open(f, "w").write(s)
pf = "/project/anon/rebuttal/port/LG-CoTeaching/pipeline.sh"
p = open(pf).read()
if "--self_training" not in p:
    p = re.sub(r'(python[^\n]*create_co_teaching_data\.py)', r'\1 --self_training ${ST:-0}', p, count=1)
    open(pf, "w").write(p)
print("PATCH_OK cctd_refs=%d pipe_refs=%d" % (s.count("self_training"), p.count("self_training")))
