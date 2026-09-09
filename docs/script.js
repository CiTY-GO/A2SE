const dimensionData = {
  plan: {
    symbol: "PLAN",
    name: "规划与探索",
    description: "负责目标分解、搜索顺序和子目标调度，解决 Agent 不知道先做什么、去哪里找的问题。",
    example: "先枚举候选位置，按空间顺序逐一搜索；已确认为空的位置不重复访问。",
    color: "coral"
  },
  track: {
    symbol: "TRACK",
    name: "状态追踪",
    description: "持续维护对象位置、属性、计数器和任务进度，避免长程交互中忘记已经做过什么。",
    example: "维护 NEEDED、HELD、PLACED 三个计数，并在每次动作后更新对象状态。",
    color: "gold"
  },
  exec: {
    symbol: "EXEC",
    name: "执行细节",
    description: "描述精确动作格式、工具使用顺序和操作前置条件，解决策略正确但动作落地错误的问题。",
    example: "到达冰箱后依次 open、put、cool、take，并确认状态变化后再进入下一阶段。",
    color: "blue"
  },
  guard: {
    symbol: "GUARD",
    name: "错误防护",
    description: "明确禁止项、恢复动作和常见失败边界，让 Agent 知道哪些看似合理的捷径不能走。",
    example: "未确认物体完成冷却前，不得提前把它放到最终目标位置。",
    color: "green"
  }
};

const symbolColors = {
  coral: ["#fff0ea", "#78331d"],
  gold: ["#fff7df", "#76520c"],
  blue: ["#eaf0fb", "#244f9f"],
  green: ["#edf5eb", "#35643c"]
};

document.querySelectorAll(".dimension-tab").forEach((button) => {
  button.addEventListener("click", () => {
    const data = dimensionData[button.dataset.dim];
    document.querySelectorAll(".dimension-tab").forEach((tab) => {
      const active = tab === button;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });

    document.querySelector("#dimension-symbol").textContent = data.symbol;
    document.querySelector("#dimension-name").textContent = data.name;
    document.querySelector("#dimension-description").textContent = data.description;
    document.querySelector("#dimension-example").textContent = data.example;
    const [background, foreground] = symbolColors[data.color];
    const symbol = document.querySelector("#dimension-symbol");
    symbol.style.background = background;
    symbol.style.color = foreground;
  });
});

const stageData = [
  {
    max: 0.5,
    label: "Stage I · 基础探索",
    active: ["plan"],
    explanation: "成功率仍低，失败多来自目标分解和探索不足。本阶段优先更新 D-PLAN。"
  },
  {
    max: 0.75,
    label: "Stage II · 状态与执行",
    active: ["track", "exec"],
    explanation: "Agent 已能完成部分任务，瓶颈转向状态丢失和动作序列。本阶段聚焦 D-TRACK 与 D-EXEC。"
  },
  {
    max: 1,
    label: "Stage III · 精细执行与防错",
    active: ["exec", "guard"],
    explanation: "基础策略趋于成熟，剩余错误集中在执行细节和约束违反。本阶段聚焦 D-EXEC 与 D-GUARD。"
  }
];

const abilitySlider = document.querySelector("#ability-slider");
const updateStage = () => {
  const value = Number(abilitySlider.value) / 100;
  const stage = stageData.find((item) => value < item.max) || stageData[2];
  document.querySelector("#ability-value").textContent = value.toFixed(2);
  document.querySelector("#stage-label").textContent = stage.label;
  document.querySelector("#stage-explanation").textContent = stage.explanation;
  document.querySelectorAll(".focus-chip").forEach((chip) => {
    chip.classList.toggle("active", stage.active.includes(chip.dataset.focus));
  });
};
abilitySlider.addEventListener("input", updateStage);
updateStage();

const benchmarkData = {
  alfworld: {
    rows: [
      ["Origin", 12.5],
      ["Mem0 + GRPO", 54.7],
      ["SimpleMem + GRPO", 62.5],
      ["GRPO", 75.0],
      ["SkillRL", 77.3],
      ["A²SE", 88.3]
    ],
    note: "A²SE 达到 88.3%，较最强技能增强基线 SkillRL 提升 11.0 pp，较标准 GRPO 提升 13.3 pp。"
  },
  webshop: {
    rows: [
      ["Origin", 3.9],
      ["Mem0 + GRPO", 37.5],
      ["SimpleMem + GRPO", 46.9],
      ["SkillRL", 71.1],
      ["GRPO", 72.6],
      ["A²SE", 79.7]
    ],
    note: "A²SE 成功率达到 79.7%，较最强列示训练基线 GRPO 提升 7.1 pp，较 SkillRL 提升 8.6 pp；任务得分达到 91.2。"
  }
};

const ablationData = {
  alfworld: [
    ["A²SE full", 88.3],
    ["w/o CASR", 84.4],
    ["w/o Stage-Aware Focus", 82.0],
    ["w/o Causal Verification", 81.3],
    ["w/o Four-Dim", 79.7],
    ["w/o Dynamic Evolution", 74.2],
    ["w/o Skill Library", 60.9]
  ],
  webshop: [
    ["A²SE full", 79.7],
    ["w/o CASR", 75.8],
    ["w/o Stage-Aware Focus", 72.7],
    ["w/o Four-Dim", 68.8],
    ["w/o Dynamic Evolution", 65.5],
    ["w/o Causal Verification", 62.5],
    ["w/o Skill Library", 50.8]
  ]
};

function renderBars(container, rows) {
  container.replaceChildren();
  rows.forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = `chart-row${label.startsWith("A²SE") ? " is-a2se" : ""}`;

    const name = document.createElement("div");
    name.className = "chart-label";
    name.textContent = label;

    const track = document.createElement("div");
    track.className = "chart-track";
    const bar = document.createElement("div");
    bar.className = "chart-bar";
    bar.style.width = `${value}%`;
    track.appendChild(bar);

    const score = document.createElement("div");
    score.className = "chart-value";
    score.textContent = value.toFixed(1);

    row.append(name, track, score);
    container.appendChild(row);
  });
}

function setToggleState(selector, activeButton) {
  document.querySelectorAll(selector).forEach((button) => {
    const active = button === activeButton;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

const resultChart = document.querySelector("#result-chart");
renderBars(resultChart, benchmarkData.alfworld.rows);
document.querySelectorAll(".chart-toggle").forEach((button) => {
  button.addEventListener("click", () => {
    setToggleState(".chart-toggle", button);
    const selected = benchmarkData[button.dataset.benchmark];
    renderBars(resultChart, selected.rows);
    document.querySelector("#result-note").textContent = selected.note;
  });
});

const ablationChart = document.querySelector("#ablation-chart");
renderBars(ablationChart, ablationData.alfworld);
document.querySelectorAll(".ablation-toggle").forEach((button) => {
  button.addEventListener("click", () => {
    setToggleState(".ablation-toggle", button);
    renderBars(ablationChart, ablationData[button.dataset.benchmark]);
  });
});
