PYTHON ?= python3

.PHONY: demo_copy demo_concat smoke_train figs eval_order eval_order_inference eval_order_noise eval_order_noise_inference eval_noise eval_noise_inference

export PYTHONHASHSEED ?= 0

runs:
	mkdir -p runs

demo_copy: runs
	$(PYTHON) runners/run_softmax_copy.py --seed 0 --out runs/demo_copy

demo_concat: runs
	$(PYTHON) runners/run_concat_inputs_targets.py --seed 0 --out runs/demo_concat

smoke_train:
	bash scripts/smoke.sh

figs:
	$(PYTHON) -m plots.attention_quick --run runs/demo_concat --out runs/demo_concat/figs

eval_order:
	# $(PYTHON) eval_order.py --models lsa1,twolayer --ordering random,smallnorm2largenorm --seeds 0 --out results_order.csv --runs-dir runs/eval_order
	# $(PYTHON) -m eval_order --models lsa1,twolayer --ordering random,smallnorm2largenorm --seeds 0 --out results_order.csv --runs-dir runs/eval_order
	$(PYTHON) -m eval_order --models lsa1,twolayer --ordering random,smallnorm2largenorm,easy2hard,hard2easy --seeds 0 --out results_order.csv --runs-dir runs/eval_order

eval_order_inference:
	# $(PYTHON) -m eval_order --models lsa1,twolayer --ordering random,smallnorm2largenorm --train_ordering random --eval_ordering auto --seeds 0 --out results_order_inference.csv --runs-dir runs/eval_order_inference
	# $(PYTHON) -m eval_order --models lsa1,twolayer --ordering random,smallnorm2largenorm --train_ordering random --eval_ordering auto --seeds 0 --out results_order_inference.csv --runs-dir runs/eval_order_inference
	$(PYTHON) -m eval_order --models lsa1,twolayer --ordering random,smallnorm2largenorm,easy2hard,hard2easy --train_ordering random --eval_ordering auto --seeds 0 --out results_order_inference.csv --runs-dir runs/eval_order_inference

# eval_order_noise:
# 	$(PYTHON) -m eval_order --models lsa1,twolayer --ordering easy2hard,hard2easy \
# 		--train_ordering auto --eval_ordering auto \
# 		--noise_mode label_noise --p 0.25 --sigma 0.5 --placement mixed \
# 		--seeds 0 --out results_order_noise.csv --runs-dir runs/eval_order_noise

eval_order_noise:
	$(PYTHON) -m eval_order --models lsa1,twolayer --ordering easy2hard,hard2easy \
		--train_ordering random --eval_ordering auto \
		--noise_mode label_noise --noise_p 0.25 --noise_sigma 0.5 --noise_placement mixed \
		--seeds 0 --out results_order_noise.csv --runs-dir runs/eval_order_noise

eval_order_noise_inference:
	$(PYTHON) -m eval_order --models lsa1,twolayer --ordering easy2hard,hard2easy \
		--train_ordering random --eval_ordering auto \
		--noise_mode label_noise --noise_p 0.25 --noise_sigma 0.5 --noise_placement mixed \
		--train_noise_mode clean --eval_noise_mode auto \
		--seeds 0 --out results_order_noise_inference.csv --runs-dir runs/eval_order_noise_inference

# NOISE EXPERIMENTS
eval_noise:
	# $(PYTHON) eval_noise.py --models lsa1,twolayer --noise_mode label_noise --p 0.0,0.25 --sigma 0.5 --placement clean_first,noisy_first --ordering random --seeds 0 --out results_noise.csv --runs-dir runs/eval_noise
	$(PYTHON) -m eval_noise --models lsa1,twolayer --noise_mode label_noise --p 0.0,0.25 --sigma 0.5 --placement clean_first,noisy_first --ordering random --seeds 0 --out results_noise.csv --runs-dir runs/eval_noise

eval_noise_inference:
	# $(PYTHON) -m eval_noise --models lsa1,twolayer --noise_mode label_noise --p 0.0,0.25 --sigma 0.5 --placement clean_first,noisy_first --ordering random --train_noise_mode clean --eval_noise_mode auto --seeds 0 --out results_noise_inference.csv --runs-dir runs/eval_noise_inference
	$(PYTHON) -m eval_noise --models lsa1,twolayer --noise_mode label_noise --p 0.0,0.5 --sigma 0.5 --placement mixed,noisy_first --ordering random --train_noise_mode clean --eval_noise_mode auto --seeds 0 --out results_noise_inference.csv --runs-dir runs/eval_noise_inference
