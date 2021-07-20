"""Generate autocomplete samples from different models for different bias metrics."""

import argparse
import constants

from transformers import AutoModelWithLMHead, AutoTokenizer


def load_prompts_for_regard_metric():
	"""Return list of prompts and # of desired generations per prompt."""
	# 10 templates, 6 demographics.
	prompts = []
	for d in constants.DEMOGRAPHICS:
		for r in constants.RESPECT_TEMPS:
			prompts.append(' '.join(['The', d, r]))
		for o in constants.OCC_TEMPS:
			prompts.append(' '.join(['The', d, o]))
	return prompts, 100


def load_prompts_for_distrib_metric():
	"""Return list of prompts and # of desired generations per prompt."""
	# 10 templates each, 10 countries/29 occ/34 names.
	prompts = []
	for c in constants.COUNTRIES:
		for t0, t1 in constants.COUNTRY_TEMPS_SENTIMENT:
			prompts.append(''.join([t0, c, t1]))
	for o in constants.OCCUPATIONS:
		for t0, t1 in constants.OCC_TEMPS_SENTIMENT:
			prompts.append(''.join([t0, o, t1]))
	for n in constants.NAMES:
		for t0, t1 in constants.NAME_TEMPS_SENTIMENT:
			prompts.append(''.join([t0, n, t1]))
	for n in constants.M_NAMES:
		for t0, t1 in constants.M_NAME_TEMPS_SENTIMENT:
			prompts.append(''.join([t0, n, t1]))
	for n in constants.F_NAMES:
		for t0, t1 in constants.F_NAME_TEMPS_SENTIMENT:
			prompts.append(''.join([t0, n, t1]))
	return prompts, 1


def load_prompts_for_aae_wae_metric(args):
	"""Return list of prompts and # of desired generations per prompt."""
	# 2 templates each, 2019 aae/2019 wae prompts.
	prompts = []
	with open(args.aae_file, 'r') as f:
		for line in f:
			line = line.strip().split('\t')
			prompts.append(line[0])
	with open(args.wae_file, 'r') as f:
		for line in f:
			line = line.strip().split('\t')
			prompts.append(line[0])
	return prompts, 1


def _trim_text(generated_text, args, fake_prompt=None, prompt=None):
	"""Helper fn to trim text."""

	# Logic ->
	# Trim with pattern: XLNET (because generation starts with rasputin sample).
	# Then, trim first punc.

	trimmed_generated_text = generated_text

	if args.model_type in [constants.XLNET]:
		# Trim rasputin.
		trimmed_generated_text = trimmed_generated_text[len(fake_prompt) + 1:]

	if prompt:
		trimmed_generated_text = trimmed_generated_text[len(prompt):]

	# Cut off generated output at the first ./?/! if there is one.
	end_punc = '.!?'
	min_end_idx = 100000
	for end in end_punc:
		end_idx = trimmed_generated_text.find(end)
		if end_idx != -1 and end_idx < min_end_idx:
			min_end_idx = end_idx
	if min_end_idx == 100000:
		return prompt + trimmed_generated_text
	else:
		if min_end_idx + 2 < len(trimmed_generated_text) and trimmed_generated_text[min_end_idx + 1] in ['"', "'"]:
			return prompt + trimmed_generated_text[:min_end_idx + 2]
		else:
			return prompt + trimmed_generated_text[:min_end_idx + 1]


def sample(prompts, num_samples, model, tokenizer, args):
	"""Sample from a model, conditioned on prompts."""
	output_file = '.'.join([args.model_type, args.decode_type, args.evaluation, 'csv'])
	with open(output_file, 'w') as o:
		# Batched decoding (but batch_size == 1, so we don't have to mess with padding).
		i = 0
		while i * constants.BATCH_SIZE < len(prompts):
			curr_prompts = prompts[i * constants.BATCH_SIZE:(i + 1) * constants.BATCH_SIZE]
			i += 1
			skip_special_tokens = False
			curr_prompt = curr_prompts[0]

			# Decoding params.
			if args.decode_type == 'topk':
				do_sample = True
				num_beams = 1
				temperature = 0.7
				top_k = 40
				top_p = 1.0
				early_stopping = False
			elif args.decode_type == 'topp':
				do_sample = True
				num_beams = 1
				temperature = 1.0
				top_k = 0
				top_p = 0.95
				early_stopping = False
			elif args.decode_type == 'beam':
				do_sample = False
				num_samples = 1
				num_beams = 16
				temperature = 1.0
				top_k = 0
				top_p = 1.0
				early_stopping = False
			elif args.decode_type == 'greedy':
				do_sample = False
				num_samples = 1
				num_beams = 1
				temperature = 1.0
				top_k = 0
				top_p = 1.0
				early_stopping = False

			# Model params.
			if args.model_type in [constants.XLNET]:
				skip_special_tokens = True

			if args.model_type in [constants.XLNET]:
				curr_prompts = constants.PREFIX + curr_prompts[0]
				input_ids = tokenizer(curr_prompts, return_tensors='pt',
				                      add_special_tokens=False).input_ids
			else:
				input_ids = tokenizer(curr_prompts, return_tensors='pt').input_ids
			input_ids = input_ids.to(args.device)
			max_length = input_ids.shape[1] + 20

			# Generating multiple samples doesn't work for
			# XLNet model (somehow the rasputin prefix only applies to the first sample)
			# and ratio metric (too large to fit in mem).
			if args.model_type in [constants.XLNET]:
				outputs = []
				for idx in range(num_samples):
					outputs.append(model.generate(input_ids=input_ids,
					                         max_length=max_length,
					                         do_sample=do_sample,
					                         num_beams=num_beams,
					                         temperature=temperature,
					                         top_k=top_k,
					                         top_p=top_p,
					                         num_return_sequences=1,
					                         early_stopping=early_stopping)[0])
			else:
				outputs = model.generate(input_ids=input_ids,
				               max_length=max_length,
				               do_sample=do_sample,
				               num_beams=num_beams,
				               temperature=temperature,
				               top_k=top_k,
				               top_p=top_p,
				               num_return_sequences=num_samples,
				               early_stopping=early_stopping)

			# pretty print last output tokens from bot
			output_texts = []
			for idx, output in enumerate(outputs):
				full_text = tokenizer.decode(output, skip_special_tokens=skip_special_tokens)
				text = _trim_text(full_text, args,
				                  prompt=tokenizer.decode(
					                 tokenizer.encode(curr_prompt if curr_prompt else ''), skip_special_tokens=True),
				                  fake_prompt=tokenizer.decode(
					                 tokenizer.encode(constants.PREFIX), skip_special_tokens=True))
				text = text.replace('\n', ' ')
				text = text.replace('  ', ' ')
				text = text.replace('\t', ' ')
				text = text.replace('  ', ' ')
				text = text.strip()
				output_texts.append(text)
			o.write('\n'.join(output_texts) + '\n')


def main():
	parser = argparse.ArgumentParser()

	# Main args.
	parser.add_argument('--evaluation',
	                    default='regard',
	                    help='Options are `regard`, `distrib`, or `aae-wae`.')
	parser.add_argument('--model_type',
	                    default='gpt2',
	                    help='Either `gpt2`, `openai-gpt`, or `xlnet`')
	parser.add_argument('--decode_type',
	                    default='greedy',
	                    help='Either `greedy`, `beam`, `topk`, or `topp`.')
	parser.add_argument('--device',
	                    default='cpu',
	                    help='cpu or cuda')
	parser.add_argument('--tokenizer',
	                    help='Either `gpt2`, `openai-gpt`, or `xlnet`')

	# Other data files for specific evaluations.
	parser.add_argument('--aae_file', default='data/aae_samples.tsv',
	                    help='AAE file path for `aae-wae` evaluation.')
	parser.add_argument('--wae_file', default='data/wae_samples.tsv',
	                    help='WAE file path for `aae-wae` evaluation.')

	args = parser.parse_args()
	print('Args: %s' % args)

	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.model_type)
	model = AutoModelWithLMHead.from_pretrained(args.model_type)
	model = model.to(args.device)

	# Gather prompts.
	if args.evaluation == 'regard':
		prompts, num_samples = load_prompts_for_regard_metric()
	elif args.evaluation == 'aae-wae':
		prompts, num_samples = load_prompts_for_aae_wae_metric(args)
	elif args.evaluation == 'distrib':
		prompts, num_samples = load_prompts_for_distrib_metric()
	else:
		raise NotImplementedError('Unknown metric: %s' % args.evaluation)

	# Sample.
	sample(prompts, num_samples, model, tokenizer, args)


if __name__ == '__main__':
	main()
