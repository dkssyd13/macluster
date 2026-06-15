# 빌린 맥 실험 런북 (2-Mac DP-vs-MP)

빌린 24GB 맥으로 실험을 돌릴 때 **위에서부터 순서대로** 따라가면 되는 체크리스트.
빌리는 시간이 한정적이니 ★ 표시는 빌리기 **전에** 48GB 맥에서 미리 끝내둘 것.

- **48GB 맥** = 런처 = rank0 = 큰 stage (항상 먼저 시작)
- **24GB 맥(빌린 것)** = 조인 = rank1 = 작은 stage
- 돌아가는 실험: `smoke`(연결확인) → `mp_mid`+`dp_mid`(DP-vs-MP 비교, gpt2_untied ~163M) → `xl`(gpt_xl 1.6B) → `3b`(gpt3b 2.78B, MP만 가능한 헤드라인)

---

## 0. ★ 빌리기 전에 (48GB 맥에서)

- [ ] **코드 최신화 + 푸시 확인.** 두 맥이 같은 커밋이어야 함.
  ```bash
  cd <repo>
  git push origin main          # 이미 79da5d1까지 푸시됨
  git rev-parse --short HEAD     # 예: 79da5d1 — 빌린 맥에서도 이 값이어야 함
  ```
- [ ] **데이터 미리 받기(pre-warm).** wikitext + tiktoken 캐시를 채워둠.
  ```bash
  uv run python -c "from macluster.data.text import make_text_task as m; t=m(1,variant='wikitext',data_dir='data/cache',seed=0); print('source',t.meta['bpe_source'],'tokens',t.meta['n_tokens'])"
  # 반드시 source=wikitext-2-raw 가 떠야 함 (tinyshakespeare-bpe-fallback 이면 네트워크 고치고 재시도)
  ```
- [ ] **데이터 캐시 복사 준비.** `data/cache/text/` 폴더(약 12MB)를 빌린 맥에 그대로 넣을 것 → 두 맥이 **똑같은 데이터**로 학습 (다르면 결과가 조용히 망가짐). 4번에서 AirDrop으로 전달.

---

## 1. 빌린 맥 점검 (받자마자)

- [ ] **칩/OS 확인.** Apple Silicon + macOS 14(Sonoma) 이상이어야 함 (mlx-metal 휠 하한).
  ```bash
  uname -m            # arm64 여야 함
  sw_vers -productVersion   # 14.x 이상
  ```
  → Intel 맥이거나 macOS 13 이하면 **이 프로젝트가 안 돌아감.** OS 업데이트하거나 다른 맥을 빌릴 것.

---

## 2. 빌린 맥 세팅 (빈 맥 기준)

- [ ] **개발 도구(git, swiftc).** grove의 AWDL 통신 헬퍼 컴파일에 `swiftc`가 필요함.
  ```bash
  xcode-select --install        # 창이 뜨면 설치 (이미 있으면 건너뜀)
  ```
- [ ] **uv 설치.**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # 새 터미널을 열거나: source $HOME/.local/bin/env
  ```
- [ ] **코드 받기 + 의존성 설치.** (uv가 Python 3.12까지 알아서 받음)
  ```bash
  git clone https://github.com/dkssyd13/macluster.git
  cd macluster
  git rev-parse --short HEAD     # 0번의 48GB 맥과 같은 값(79da5d1)인지 확인
  uv sync                        # 첫 설치라 인터넷 필요 (~70MB)
  ```

---

## 3. ★ 데이터 캐시 심기 (빌린 맥)

- [ ] 48GB 맥의 `data/cache/text/` 폴더를 빌린 맥의 `<repo>/data/cache/text/`로 복사(AirDrop).
  - 캐시가 있으면 빌린 맥이 다시 다운로드하지 않아 **두 맥의 데이터 바이트가 동일**해짐.
  - 안 했어도 자동 다운로드되지만, 다른 시점에 받으면 미세하게 달라질 위험 → 가급적 복사할 것.
- [ ] 확인:
  ```bash
  uv run python -c "from macluster.data.text import make_text_task as m; t=m(1,variant='wikitext',data_dir='data/cache',seed=0); print(t.meta['bpe_source'], t.meta['n_tokens'])"
  # 48GB 맥과 source / tokens 가 똑같이 나와야 함
  ```

---

## 4. 네트워크 연결 (두 맥 모두)

- [ ] **두 맥 모두 Finder → AirDrop 창을 열고 "모든 사람"으로** 설정 → awdl0 무선 인터페이스 활성화. (grove는 AWDL P2P로 서로를 찾음. 캠퍼스 Wi-Fi 불필요.)
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

**그다음 전체 실행 (또는 원하는 페이즈만):**
```bash
# 48GB 맥
./scripts/run_mac_48gb.sh                 # smoke mp_mid dp_mid xl 3b 전부
# 24GB 맥
./scripts/run_mac_24gb.sh
```
- 헤드라인 비교만 빨리: 양쪽에서 `./scripts/run_mac_*.sh mp_mid dp_mid`
- `3b`가 OOM 나면 `configs/grove/pipeline_3b.env`에서 `MACLUSTER_BATCH_SIZE`를 4→2로 낮춤. 앞 페이즈 결과는 그대로 남음.
- 문제가 생기면 한 페이즈씩 따로 돌리는 게 안전 (페이즈 어긋남 방지).

---

## 6. ★★ 결과 회수 (반납 전 필수)

손실/perplexity/메모리 같은 **헤드라인 숫자는 rank1(빌린 맥)에만** 저장됨. 반납 전에 반드시 48GB 맥으로 옮길 것 — **안 옮기면 영구 손실.**

- [ ] 48GB 맥에서 원격 로그인 켜기: 시스템 설정 → 일반 → 공유 → **원격 로그인** ON. 48GB 스크립트가 끝나면 정확한 `RESULTS_DEST=...` 값을 출력해줌.
- [ ] 빌린 맥에서 그 값으로 결과 전송:
  ```bash
  RESULTS_DEST=<사용자>@<48GB맥IP>:<repo경로>/runs/  ./scripts/run_mac_24gb.sh
  # 또는 이미 실험이 끝났다면 runs/ 폴더를 통째로 AirDrop
  ```
- [ ] **복사 성공 확인** (48GB 맥에서):
  ```bash
  ls runs/*-rank1/summary.json     # rank1 결과가 들어왔는지
  ```
  → rank1 summary가 보이고 손실/perplexity가 들어있으면 OK. 그제서야 빌린 맥 반납.

---

## 결과가 만들어내는 것 (논문용)

- `runs/*mac_mp*-rank{0,1}/` + `runs/*mac_dp*-rank{0,1}/` → **DP vs MP 정면 비교** (gpt2_untied, 같은 데이터량): 벽시계 시간, 통신량(`total_comm_MB`), 단계/복제별 peak 메모리, 수렴(val_loss/perplexity).
- `runs/*mac_3b*-rank{0,1}/` → **메모리월 헤드라인**: MP는 30GB/15GB로 쪼개져 돌아가지만, 같은 모델의 풀 복제 ~44GB는 24GB 노드에 안 들어가 DP 불가.
- 비교 그림은 반납 후 48GB 맥에서 `runs/`의 `summary.json`/`metrics.jsonl`을 읽어 그리면 됨 (peak_mem / comm / wall-clock을 parallelism별로 묶기 — 이 플롯 스크립트는 아직 없으니 필요하면 추가 요청).
