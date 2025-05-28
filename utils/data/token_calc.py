import os
import csv

def compute_tokenization_statistics(path, tok_type='bpe'):
    if not os.path.exists(path):
        return {
            'avg_tokens_per_sentence': 0.0,
            'avg_tokens_per_word': 0.0,
            'avg_words_per_sentence': 0.0,
            'split_word_rate': 0.0,
            'total_sentences': 0,
        }

    total_sentences = 0
    total_tokens = 0
    total_words = 0
    total_split_words = 0

    with open(path, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            total_sentences += 1
            total_tokens += len(tokens)

            i = 0
            while i < len(tokens):
                token = tokens[i]

                if tok_type.startswith('bpe'):
                    current_word_token_count = 1
                    while token.endswith('@@') and (i + 1) < len(tokens):
                        i += 1
                        token = tokens[i]
                        current_word_token_count += 1
                    total_words += 1
                    if current_word_token_count > 1:
                        total_split_words += 1

                elif tok_type.startswith('spm'):
                    if token.startswith('▁'):
                        total_words += 1
                        # Check continuation
                        j = i + 1
                        while j < len(tokens) and not tokens[j].startswith('▁'):
                            j += 1
                        if j - i > 1:
                            total_split_words += 1

                else:
                    raise ValueError(f'Unsupported tokenization type: {tok_type}')

                i += 1

    return {
        'avg_words_per_sentence': round(total_words/total_sentences, 2) if total_sentences > 0 else 0.0,
        'avg_tokens_per_sentence': round(total_tokens/total_sentences, 2) if total_sentences > 0 else 0.0,
        'avg_tokens_per_word': round(total_tokens/total_words, 2) if total_words > 0 else 0.0,
        'split_word_rate': round(total_split_words/total_words * 100, 2) if total_words > 0 else 0.0,
        'total_sentences': total_sentences
    }

def main():
    directory = '/afs/crc.nd.edu/group/nlp/data/mbrmt/'
    tokenizations = ['bpe', 'bpe_drp', 'spm_bpe', 'spm_uni']
    target_languages = ['az', 'is', 'ja', 'la', 'tk', 'tr', 'uz']
    results = []
    
    for tok in tokenizations:
        for lang in target_languages:
            for side in ['en', lang]:
                path = directory + 'en-' + lang + '_' + tok + '/' + 'train/train.norm.tok.' + tok.split('_')[0] + '.' + side
                metrics = compute_tokenization_statistics(path, tok)
                results.append({
                        'language': lang,
                        'side': 'en' if side == 'en' else 'target',
                        'tokenization': tok,
                        **metrics
                    })

    output_csv = 'tokenization_analysis.csv'
    fieldnames = ['language', 'side', 'tokenization',
                  'avg_words_per_sentence', 'avg_tokens_per_sentence', 'avg_tokens_per_word', 'split_word_rate', 'total_sentences']
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print('done')

if __name__ == '__main__':
    main()
