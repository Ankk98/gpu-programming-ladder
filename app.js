// Ladder data model key
const LADDER_KEY = 'gpuProgrammingLadderProgress_v1';

// Status options
const STATUS = {
  NOT_STARTED: 'not-started',
  IN_PROGRESS: 'in-progress',
  COMPLETED: 'completed',
};

const statusLabels = {
  [STATUS.NOT_STARTED]: 'Not started',
  [STATUS.IN_PROGRESS]: 'In progress',
  [STATUS.COMPLETED]: 'Completed',
};

// Load/save progress from localStorage
function loadProgress() {
  try {
    const raw = localStorage.getItem(LADDER_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (e) {
    return {};
  }
}

function saveProgress(progress) {
  try {
    localStorage.setItem(LADDER_KEY, JSON.stringify(progress));
  } catch (e) {
    console.warn('Unable to save progress', e);
  }
}

const progress = loadProgress();

function getTopicStatus(phaseId, topicId) {
  const key = `${phaseId}:${topicId}`;
  return progress[key] || STATUS.NOT_STARTED;
}

function setTopicStatus(phaseId, topicId, status) {
  const key = `${phaseId}:${topicId}`;
  progress[key] = status;
  saveProgress(progress);
}

function computePhaseStats(phase) {
  let total = 0;
  let completed = 0;
  let inProgress = 0;
  phase.groups.forEach((g) => {
    g.topics.forEach((t) => {
      total += 1;
      const s = getTopicStatus(phase.id, t.id);
      if (s === STATUS.COMPLETED) completed += 1;
      else if (s === STATUS.IN_PROGRESS) inProgress += 1;
    });
  });
  const percent =
    total === 0 ? 0 : Math.round((completed / total) * 100);
  return { total, completed, inProgress, percent };
}

// Render phase list in sidebar
const phaseListEl = document.getElementById('phaseList');
const phaseNameEl = document.getElementById('phaseName');
const phaseTagEl = document.getElementById('phaseTag');
const phaseDescEl = document.getElementById('phaseDesc');
const phaseProgressFillEl = document.getElementById('phaseProgressFill');
const phaseProgressSummaryEl =
  document.getElementById('phaseProgressSummary');
const phaseProgressPercentEl =
  document.getElementById('phaseProgressPercent');
const topicsContainerEl = document.getElementById('topicsContainer');
const resetBtn = document.getElementById('resetProgressBtn');
const markPhaseCompleteBtn =
  document.getElementById('markPhaseCompleteBtn');

let currentPhaseId = ladder[0]?.id ?? null;

function renderPhaseList() {
  phaseListEl.innerHTML = '';
  ladder.forEach((phase) => {
    const stats = computePhaseStats(phase);
    const li = document.createElement('li');
    const item = document.createElement('button');
    item.type = 'button';
    const isActive = phase.id === currentPhaseId;
    item.className = `w-full text-left px-2.5 py-2 rounded-lg border transition-all flex items-center justify-between gap-2 text-xs ${isActive
        ? 'border-primary-500 bg-primary-500/10 shadow-sm'
        : 'border-transparent bg-gray-50 dark:bg-gray-800/40 hover:border-gray-300 dark:hover:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800'
      }`;
    item.dataset.phaseId = phase.id;

    const left = document.createElement('div');
    left.className = 'flex flex-col items-start gap-0.5';

    const label = document.createElement('div');
    label.className = 'font-medium';
    label.textContent = phase.name.replace(/^Phase \d+ – /, '');
    left.appendChild(label);

    const meta = document.createElement('div');
    meta.className = 'text-xs text-gray-600 dark:text-gray-400';
    meta.textContent = `${stats.completed}/${stats.total} complete`;
    left.appendChild(meta);

    const right = document.createElement('div');
    right.className = 'text-xs px-2 py-0.5 rounded-full border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 whitespace-nowrap';
    right.textContent = `${stats.percent}%`;

    item.appendChild(left);
    item.appendChild(right);
    li.appendChild(item);
    phaseListEl.appendChild(li);

    item.addEventListener('click', () => {
      currentPhaseId = phase.id;
      renderPhaseList();
      renderCurrentPhase();
    });
  });
}

function renderCurrentPhase() {
  const phase = ladder.find((p) => p.id === currentPhaseId) || ladder[0];
  if (!phase) return;
  phaseNameEl.textContent = phase.name;
  phaseTagEl.textContent = phase.tag;
  phaseDescEl.textContent = phase.description;

  const stats = computePhaseStats(phase);
  phaseProgressFillEl.style.width = `${stats.percent}%`;
  phaseProgressSummaryEl.textContent =
    `${stats.completed}/${stats.total} topics completed · ` +
    `${stats.inProgress} in progress`;
  phaseProgressPercentEl.textContent = `${stats.percent}%`;

  topicsContainerEl.innerHTML = '';

  phase.groups.forEach((group) => {
    const groupEl = document.createElement('div');
    groupEl.className = 'topic-group p-3.5 rounded-xl bg-white/90 dark:bg-gray-800/90 border border-gray-200 dark:border-gray-700 shadow-sm';
    groupEl.dataset.groupId = group.id;

    const headerEl = document.createElement('div');
    headerEl.className = 'topic-group-header flex justify-between items-center gap-2 cursor-pointer';

    const textBox = document.createElement('div');
    textBox.className = 'flex flex-col gap-0.5';

    const titleEl = document.createElement('div');
    titleEl.className = 'text-sm font-medium flex items-center gap-2';
    titleEl.innerHTML = `<span>${group.title}</span>`;

    const metaEl = document.createElement('div');
    metaEl.className = 'text-xs text-gray-600 dark:text-gray-400';
    metaEl.textContent = group.meta;

    textBox.appendChild(titleEl);
    textBox.appendChild(metaEl);

    const toggleEl = document.createElement('div');
    toggleEl.className = 'topic-group-toggle text-lg leading-none text-gray-600 dark:text-gray-400';
    toggleEl.textContent = '▾';

    headerEl.appendChild(textBox);
    headerEl.appendChild(toggleEl);

    const listEl = document.createElement('div');
    listEl.className = 'topic-list mt-2.5 flex flex-col gap-2';

    group.topics.forEach((topic) => {
      const topicEl = document.createElement('div');
      topicEl.className = 'p-2.5 rounded-lg border border-gray-200 dark:border-gray-700 bg-white/90 dark:bg-gray-800/90 grid grid-cols-[auto_1fr] gap-2 items-start text-xs sm:grid-cols-1';

      const status = getTopicStatus(phase.id, topic.id);

      const mainEl = document.createElement('div');
      mainEl.className = 'flex flex-col gap-1';

      const titleRow = document.createElement('div');
      titleRow.className = 'flex gap-1.5 items-center flex-wrap';

      const title = document.createElement('div');
      title.className = 'font-medium text-sm';
      title.textContent = topic.title;
      titleRow.appendChild(title);

      mainEl.appendChild(titleRow);

      if (topic.description) {
        const desc = document.createElement('div');
        desc.className = 'text-xs text-gray-600 dark:text-gray-400';
        desc.textContent = topic.description;
        mainEl.appendChild(desc);
      }

      const linksEl = document.createElement('div');
      linksEl.className = 'flex flex-wrap gap-1.5 items-center text-xs';

      const createLinkPill = (href, icon, text, isMissing) => {
        const pill = document.createElement('a');
        if (isMissing) {
          pill.className = 'px-2 py-1 rounded-full border border-dashed border-orange-500/60 text-orange-500 bg-orange-500/5 inline-flex items-center gap-1.5 no-underline';
          pill.href = '#';
          pill.addEventListener('click', (e) => e.preventDefault());
        } else {
          pill.className = 'px-2 py-1 rounded-full border border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 inline-flex items-center gap-1.5 no-underline';
          pill.href = href;
          pill.target = '_blank';
          pill.rel = 'noopener noreferrer';
        }
        pill.innerHTML = `<span class="w-3 h-3 rounded-full bg-gray-200 dark:bg-gray-700 text-[9px] flex items-center justify-center">${icon}</span><span>${text}</span>`;
        return pill;
      };

      linksEl.appendChild(createLinkPill(
        topic.article,
        'A',
        topic.article ? 'Article' : 'Article: needs better resource',
        !topic.article
      ));
      linksEl.appendChild(createLinkPill(
        topic.video,
        '▶',
        topic.video ? 'Video' : 'Video: needs better resource',
        !topic.video
      ));
      linksEl.appendChild(createLinkPill(
        topic.paper,
        '📄',
        topic.paper ? 'Paper' : 'Paper: needs seminal reference',
        !topic.paper
      ));
      linksEl.appendChild(createLinkPill(
        topic.exercise,
        '⚑',
        topic.exercise ? 'Exercise' : 'Exercise: needs better resource',
        !topic.exercise
      ));

      if (topic.python) {
        linksEl.appendChild(createLinkPill(
          topic.python,
          '🐍',
          'Python',
          false
        ));
      }

      if (topic.cpp) {
        linksEl.appendChild(createLinkPill(
          topic.cpp,
          '⚙',
          'C++',
          false
        ));
      }

      mainEl.appendChild(linksEl);

      const statusWrapper = document.createElement('div');
      const select = document.createElement('select');
      const statusClasses = {
        [STATUS.COMPLETED]: 'border-emerald-500/70 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
        [STATUS.IN_PROGRESS]: 'border-amber-500/70 bg-amber-500/10 text-amber-600 dark:text-amber-400',
        [STATUS.NOT_STARTED]: 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-400',
      };
      select.className = `min-w-[110px] px-2 py-1 rounded-full border text-xs outline-none cursor-pointer ${statusClasses[status] || statusClasses[STATUS.NOT_STARTED]}`;
      select.dataset.phaseId = phase.id;
      select.dataset.topicId = topic.id;

      Object.values(STATUS).forEach((value) => {
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = statusLabels[value];
        if (value === status) opt.selected = true;
        select.appendChild(opt);
      });

      statusWrapper.appendChild(select);

      topicEl.appendChild(statusWrapper);
      topicEl.appendChild(mainEl);

      listEl.appendChild(topicEl);

      select.addEventListener('change', () => {
        const newStatus = select.value;
        setTopicStatus(phase.id, topic.id, newStatus);
        select.className = `min-w-[110px] px-2 py-1 rounded-full border text-xs outline-none cursor-pointer ${statusClasses[newStatus] || statusClasses[STATUS.NOT_STARTED]}`;
        renderPhaseList();
        renderCurrentPhase();
      });
    });

    groupEl.appendChild(headerEl);
    groupEl.appendChild(listEl);

    // Collapsing behavior
    groupEl.classList.add('expanded');
    headerEl.addEventListener('click', () => {
      groupEl.classList.toggle('collapsed');
    });

    topicsContainerEl.appendChild(groupEl);
  });
}

// Reset progress
resetBtn.addEventListener('click', () => {
  if (
    !window.confirm(
      'Reset all progress for GPU Programming Ladder in this browser?'
    )
  ) {
    return;
  }
  Object.keys(progress).forEach((k) => {
    delete progress[k];
  });
  saveProgress(progress);
  renderPhaseList();
  renderCurrentPhase();
});

// Mark current phase as completed
markPhaseCompleteBtn.addEventListener('click', () => {
  const phase = ladder.find((p) => p.id === currentPhaseId);
  if (!phase) return;
  phase.groups.forEach((g) => {
    g.topics.forEach((t) => {
      setTopicStatus(phase.id, t.id, STATUS.COMPLETED);
    });
  });
  renderPhaseList();
  renderCurrentPhase();
});

// Initial render
renderPhaseList();
renderCurrentPhase();
