# 빌린 맥 실험 런북 (2-Mac DP-vs-MP)

빌린 24GB 맥으로 실험을 돌릴 때 **위에서부터 순서대로** 따라가는 체크리스트.
빌리는 시간이 한정적이니 ★ 표시는 빌리기 **전에** 48GB 맥에서 미리 끝내둘 것.

- **48GB 맥** = 런처 = rank0 = 큰 stage (항상 먼저 시작)
- **24GB 맥(빌린 것)** = 조인 = rank1 = 작은 stage
- 페이즈: `smoke`(연결확인) → `mp_mid`+`dp_mid`(DP-vs-MP 비교, gpt2_untied ~163M) → `xl`(gpt_xl 1.6B, 3b 예행) → `3b`(gpt3b 2.78B, **MP만 가능한 헤드라인**)

> **검증 완료(2026-06-20):** 실제 2랭크 grove 경로(MP 파이프라인 + DP diloco)를 48GB 맥
> 한 대에서 프로세스 2개로 실 wikitext + gpt2_untied로 끝까지 돌려, **deadlock 없이
> 완주 + 보고서용 수치 전부 생성**됨을 확인했다. 78개 테스트도 전부 통과. 즉 코드
> 경로는 검증됐고, 빌린 맥에서 필요한 건 "두 대로 쪼개야만 의미가 있는" 실 벽시계 /
> 노드별 실 peak 메모리 / gpt3b 메모리월뿐이다.

---

## 0. ★ 빌리기 전에 (48GB 맥에서) — 대부분 완료됨

- [x] **코드 최신화 + 푸시.** 두 맥이 같은 커밋이어야 함. 현재 `HEAD == origin/main`.
  ```bash
  git rev-parse --short HEAD     # 예: cb90e74 — 빌린 맥에서도 이 값이어야 함
  git push origin main           # (이미 동기화돼 있으면 'up to date')
  ```
- [x] **데이터 미리 받음(pre-warm).** `data/cache/text/wikitext2_1.txt`(+`tinyshakespeare.txt`,
  합쳐 ~11MB)가 48GB 맥에 이미 있음. 확인:
  ```bash
  uv run python -c "from macluster.data.text import make_text_task as m; t=m(1,variant='wikitext',batch_size=8,seq_len=128,data_dir='data/cache',seed=0); print(t.meta['bpe_source'], t.meta['n_tokens'])"
  # 반드시 'wikitext-2-raw 2448382' 가 떠야 함 (tinyshakespeare-bpe-fallback 이면 네트워크 고치고 재시도)
  ```

---

## 1. 빌린 맥 점검 (받자마자, ~5분)

- [ ] **칩/OS 확인.** Apple Silicon + macOS 14(Sonoma) 이상 (mlx-metal 휠 하한).
  ```bash
  uname -m                  # arm64 여야 함
  sw_vers -productVersion   # 14.x 이상
  ```
  → Intel 맥이거나 macOS 13 이하면 **이 프로젝트가 안 돌아감.** OS 업데이트하거나 다른 맥을 빌릴 것.

---

## 2. 빌린 맥 세팅 (빈 맥 기준)

- [ ] **개발 도구(git, swiftc).** grove의 AWDL 통신 헬퍼 컴파일에 `swiftc`가 필요함.
  ```bash
  xcode-select --install        # 창이 뜨면 설치 (이미 있으면 자동 skip)
  ```
- [ ] **uv 설치.**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env   # 또는 새 터미널
  ```
- [ ] **코드 받기 + 의존성 설치.** (uv가 Python 3.12까지 알아서 받음)
  ```bash
  git clone https://github.com/dkssyd13/macluster.git
  cd macluster
  git rev-parse --short HEAD     # 0번의 48GB 맥과 같은 값인지 확인
  uv sync                        # 첫 설치라 인터넷 필요 (~70MB)
  ```

---

## 3. ★ 데이터 캐시 심기 (빌린 맥) — **두 맥이 같은 데이터를 봐야 함 (제일 중요)**

빌린 맥이 wikitext를 직접 받게 두면, 다운로드 URL이 가끔 실패해서 **조용히 tinyshakespeare로
대체**된다. vocab(50257)이 같아 에러는 안 나지만, rank0은 wikitext, rank1은 tinyshakespeare로
학습/평가하게 되어 **loss/perplexity가 의미 없는 값이 됨** (반납 후에야 발견). 그래서 캐시를
직접 복사한다.

- [ ] 48GB 맥의 `data/cache/text/` 폴더를 빌린 맥의 `<repo>/data/cache/text/`로 **AirDrop** 복사.
  (`_download_text`는 이미 있는 파일은 건너뛰므로, 복사해두면 빌린 맥이 다시 받지 않고 그대로 씀.)
- [ ] **두 맥 모두에서 같은 값이 뜨는지 직접 확인** (다르면 절대 본 실험 시작 금지):
  ```bash
  uv run python -c "from macluster.data.text import make_text_task as m; t=m(1,variant='wikitext',batch_size=8,seq_len=128,data_dir='data/cache',seed=0); print(t.meta['bpe_source'], t.meta['n_tokens'])"
  # 두 맥 모두 'wikitext-2-raw 2448382' 가 똑같이 떠야 함
  ```
  > **자동 안전장치(코드).** 위 수동 확인을 깜빡해도, 본 실험 시작 시 두 맥이 다른 코퍼스를
  > 들었으면 `assert_data_consensus`가 랭크 간 데이터 지문(corpus/source/token수/vocab)을
  > 대조해 **즉시 중단**시킨다 (`grove_backend.py`). 그래도 캐시 복사가 1차 방어선이니 꼭 할 것.

---

## 4. 네트워크 연결 (두 맥 모두)

- [ ] **두 맥 모두 Finder → AirDrop 창을 열고 "모든 사람"으로** 설정 → awdl0 무선 인터페이스 활성화.
  (grove는 AWDL P2P로 서로를 찾음. 캠퍼스 Wi-Fi 불필요.)
- [ ] 서로 보이는지 확인 (한쪽이 start 상태일 때):
  ```bash
  uv run grove status
  ```

---

## 5. 실험 실행

> 48GB 맥을 **먼저** 시작하고, 2분 안에 24GB 맥을 시작. 같은 페이즈 인자를 양쪽에 동일하게.

**(권장) 먼저 smoke만 돌려 연결/seam 확인:**
```bash
# 48GB 맥
./scripts/run_mac_48gb.sh smoke
# 24GB 맥 (바로 이어서)
./scripts/run_mac_24gb.sh smoke
```
→ rank1(24GB)에서 손실 숫자가 찍히고 에러가 없으면 통과.

**★ 본 실험 전에 결과 회수 경로를 먼저 잡아둘 것 (세션 내내 사용):**
48GB 맥에서 시스템 설정 → 일반 → 공유 → **원격 로그인 ON**. 48GB 스크립트가 끝나면 정확한
`RESULTS_DEST=...` 값을 출력해준다. 그 값을 24GB 맥의 **첫 실행부터** 붙여서 돌린다(아래).

**그다음 전체 실행 (또는 원하는 페이즈만):**
```bash
# 48GB 맥
./scripts/run_mac_48gb.sh                 # smoke mp_mid dp_mid xl 3b 전부
# 24GB 맥  (RESULTS_DEST 를 처음부터 붙여 매 페이즈마다 자동 회수)
RESULTS_DEST=<user>@<48GB맥IP>:<repo경로>/runs/  ./scripts/run_mac_24gb.sh
```
- 헤드라인 비교만 빨리: 양쪽에서 `./scripts/run_mac_*.sh mp_mid dp_mid`
- 각 페이즈는 독립 실행(클러스터명이 페이즈마다 다름: `mac_smoke/mac_mp/mac_dp/mac_xl/mac_3b`)
  이라, 뒤 페이즈가 OOM 나도 앞 페이즈 결과는 그대로 남는다.
- **3b 전에 xl로 예행:** xl이 양쪽 맥에 여유 있게 들어가는지 확인. 메모리가 빠듯해 보이면
  `configs/grove/pipeline_3b.env`의 `MACLUSTER_BATCH_SIZE`를 8→4(→2)로 낮춘다. (param/optimizer
  state는 고정이고 activation만 줄어듦. 가장 다시 돌리기 싫은 게 3b이므로 보수적으로.)
- 페이즈마다 끝나면 48GB 맥에서 `ls runs/<슬러그>-rank1/metrics.jsonl` 로 회수됐는지 확인.

---

## 6. ★★ 결과 회수 + 반납 전 검증 (필수)

수렴치(loss/perplexity)가 어느 랭크에 저장되는지 **비대칭**임을 알아둘 것 (코드 확인 결과):
- **MP(파이프라인):** 마지막 랭크 = **rank1(빌린 맥)** 에 train_loss/val_loss/perplexity 저장
- **DP(diloco):** **rank0(48GB 맥)** 에 저장
- comm량·peak 메모리는 **양쪽 다** 저장 (per-node/per-stage)

→ MP 수렴 곡선과 gpt3b 헤드라인 loss는 **빌린 맥에만** 있으므로, **반드시 48GB 맥으로 회수**해야 함.

- [ ] **회수.** 5번처럼 `RESULTS_DEST`를 붙여 24GB 스크립트를 돌렸다면 매 페이즈 끝에 자동 rsync됨.
  스크립트는 이제 **하드 게이트**: `RESULTS_DEST`가 비었거나 rsync가 실패하면 조용히 끝나지 않고
  큰 경고 + `exit 1`로 멈춘다(깜빡 방지). 수동으로 한 번에 옮기려면 `runs/` 폴더를 통째로 AirDrop.
- [ ] **복사 성공 검증** (48GB 맥에서):
  ```bash
  ls runs/*mac_mp*-rank1/metrics.jsonl runs/*mac_dp*-rank0/metrics.jsonl runs/*mac_3b*-rank1/metrics.jsonl
  # 각 파일의 마지막 줄에 train_loss(평가 라운드면 val_loss)가 있으면 OK
  ```
  > **OOM 주의.** `summary.json`은 라운드 루프가 **끝까지** 돌아야 써진다. 3b가 OOM으로 중간에
  > 죽으면 summary.json은 없지만, `metrics.jsonl`은 매 라운드 기록되므로 **거기까지의 곡선은 살아있다.**
  > 그러니 반납 전 검증은 `summary.json`이 아니라 **`metrics.jsonl`이 비어있지 않은지**로 판단할 것.
- [ ] rank1 metrics가 48GB 맥에 들어왔고 마지막 줄에 loss가 있으면 **그제서야 빌린 맥 반납.**

---

## 결과가 만들어내는 것 (반납 후, 48GB 맥에서)

한 번의 5-페이즈 실행으로 보고서 수치가 전부 나온다 (재실행 불필요). 그림/표는 다음으로 생성:

```bash
uv run python scripts/plot_dp_vs_mp.py --runs runs --out figures/dp_vs_mp --node-mem-gb 48,24
```
→ `figures/dp_vs_mp/` 에 생성:
- `convergence.png` — DP vs MP val-loss/perplexity (vs 샘플수, vs 벽시계)
- `comm.png` — 총 통신량(MB) DP vs MP (검증 데이터에선 DP 1304MB vs MP 100MB ≈ 13배)
- `walltime.png` — 총 벽시계(s) DP vs MP
- `memory.png` — 노드/스테이지별 실측 peak 메모리 + 24/48GB 노드 한계선 (메모리월 헤드라인)
- `summary.md` / `summary.csv` — 표 (peak mem, comm, loss, wall-clock, cut/파라미터 등)

**보고서 작성 시 주의 (감사에서 나온 공정성 caveat):**
- DP-vs-MP는 **시스템 지표(벽시계·통신량·peak 메모리·메모리월)** 로만 비교할 것.
- 수렴 품질(val_loss/perplexity)은 **참고용**으로만 — 같은 3200 샘플이라도 MP(업데이트 100회,
  유효배치 32)와 DP(복제당 200회, 유효배치 8, + outer_lr=0.7)는 업데이트 수·유효배치·LR 의미가
  달라 "어느 쪽이 더 잘 수렴" 식 주장은 금물. (평가셋은 양쪽 동일한 held-out tail이라 측정 자체는 공정.)
- 헤드라인은 견고함: "gpt3b의 전체 복제 state ~44GB는 24GB 노드에 안 들어가 DP 불가 →
  메모리 인지 분할(rank0~30GB / rank1~15GB)로만 학습 가능." (사이징은 해석적, peak 메모리는 실측.)
