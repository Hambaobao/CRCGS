from .evaluate import fx_calc_map_label_with_diff_label
from .generate_embedding import generate_binary_emb, generate_emb


def cal_map_results(stu_model, query_loader, retrieval_loader, bin_model, binary=True):
    # get embedding
    query_emb = {}
    retrieval_emb = {}

    if binary:
        print('>>> generate binary embedding <<<')
        query_emb['image'], query_emb['text'], query_emb['label'] = generate_binary_emb(stu_model, bin_model, query_loader)
        retrieval_emb['image'], retrieval_emb['text'], retrieval_emb['label'] = generate_binary_emb(stu_model, bin_model, retrieval_loader)
    else:
        print('>>> generate embedding <<<')
        query_emb['image'], query_emb['text'], query_emb['label'] = generate_emb(stu_model, query_loader)
        retrieval_emb['image'], retrieval_emb['text'], retrieval_emb['label'] = generate_emb(stu_model, retrieval_loader)

    print('>>> Evaluating <<<')
    img2txt = fx_calc_map_label_with_diff_label(query_emb['image'], retrieval_emb['text'], query_emb['label'], retrieval_emb['label'], dist_method='Ham')
    txt2img = fx_calc_map_label_with_diff_label(query_emb['text'], retrieval_emb['image'], query_emb['label'], retrieval_emb['label'], dist_method='Ham')

    return img2txt, txt2img