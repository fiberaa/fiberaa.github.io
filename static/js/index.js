function copyBibTeX() {
  const code = document.getElementById("bibtex-code");
  const button = document.getElementById("copy-bibtex");
  if (!code || !button) return;

  const markCopied = () => {
    button.textContent = "Copied";
    button.classList.add("is-copied");
    window.setTimeout(() => {
      button.textContent = "Copy";
      button.classList.remove("is-copied");
    }, 1800);
  };

  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(code.textContent).then(markCopied);
    return;
  }

  const textArea = document.createElement("textarea");
  textArea.value = code.textContent;
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand("copy");
  textArea.remove();
  markCopied();
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setupGammaChart() {
  const svg = document.querySelector("[data-gamma-chart]");
  if (!svg) return;
  const ns = "http://www.w3.org/2000/svg";
  const W=900,H=210,L=54,R=16,T=14,B=38,maxK=20;
  const x=k=>L+k/maxK*(W-L-R), y=v=>T+(1-v)*(H-T-B);
  const add=(tag,attrs,text)=>{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;svg.appendChild(e);return e;};
  [0,.25,.5,.75,1].forEach(v=>{add("line",{x1:L,y1:y(v),x2:W-R,y2:y(v),stroke:"#52697a","stroke-width":".7"});add("text",{x:L-9,y:y(v)+4,fill:"#8fa5bf","font-size":"11","text-anchor":"end"},v.toFixed(2));});
  [0,4,8,12,16,20].forEach(k=>{add("line",{x1:x(k),y1:T,x2:x(k),y2:H-B,stroke:"#344956","stroke-width":".7"});add("text",{x:x(k),y:H-22,fill:"#8fa5bf","font-size":"11","text-anchor":"middle"},k===0?"sₜ":`sₜ₊${k}`);});
  add("text",{x:15,y:95,fill:"#9eb0c6","font-size":"12",transform:"rotate(-90 15 95)","text-anchor":"middle"},"weight γᵏ");
  [[.8,"#7895aa",""],[.98,"#d0d7dd","6 4"]].forEach(([g,c,d])=>{const pts=Array.from({length:maxK+1},(_,k)=>[x(k),y(Math.pow(g,k))]);add("polyline",{points:pts.map(p=>p.join(",")).join(" "),fill:"none",stroke:c,"stroke-width":"2.5","stroke-dasharray":d});pts.forEach(p=>add("circle",{cx:p[0],cy:p[1],r:"2.5",fill:"#0d161e",stroke:c,"stroke-width":"1.5"}));});
}

function setupMathFormula() {
  document.querySelectorAll("[data-katex]").forEach((element) => {
    if (!window.katex) return;
    window.katex.render(element.dataset.katex, element, {
      displayMode: true,
      throwOnError: false,
      strict: "ignore",
    });
  });
}

function loadDeferredVideo(video) {
  const useHdHero = video.matches("[data-hero-video]")
    && window.matchMedia("(min-width: 700px)").matches;
  const source = useHdHero ? (video.dataset.srcHd || video.dataset.src) : video.dataset.src;
  if (video.getAttribute("src") || !source) return;
  video.setAttribute("src", source);
  video.load();
}

function setupHeroVideoMatrix() {
  const matrix = document.querySelector(".hero-video-matrix");
  const videos = Array.from(document.querySelectorAll("[data-hero-video]"));
  if (!matrix || !videos.length) return;

  const reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  const mobileHeroQuery = window.matchMedia("(max-width: 699px)");
  const pendingPlayback = new WeakSet();
  const retryCounts = new WeakMap();
  const retryTimers = new WeakMap();
  let matrixInView = false;

  const getPlayableVideos = () => (
    mobileHeroQuery.matches
      ? videos.filter((video, index) => index % 4 < 2)
      : videos
  );

  const clearRetry = (video) => {
    const retryTimer = retryTimers.get(video);
    if (retryTimer) window.clearTimeout(retryTimer);
    retryTimers.delete(video);
    retryCounts.delete(video);
  };

  const shouldPlay = (video) => (
    matrixInView
    && !document.hidden
    && !reducedMotionQuery.matches
    && video.dataset.heroAutoplay === "true"
  );

  const scheduleRetry = (video) => {
    if (!shouldPlay(video) || retryTimers.has(video)) return;
    const retryCount = (retryCounts.get(video) || 0) + 1;
    if (retryCount > 8) return;
    retryCounts.set(video, retryCount);

    const retryTimer = window.setTimeout(() => {
      retryTimers.delete(video);
      attemptPlayback(video);
    }, Math.min(160 * (2 ** (retryCount - 1)), 1600));
    retryTimers.set(video, retryTimer);
  };

  function attemptPlayback(video) {
    if (!shouldPlay(video)) return;
    if (!video.paused && !video.ended) {
      clearRetry(video);
      return;
    }
    if (pendingPlayback.has(video)) return;

    video.autoplay = true;
    video.defaultMuted = true;
    video.muted = true;
    video.playsInline = true;
    loadDeferredVideo(video);

    let playPromise;
    try {
      playPromise = video.play();
    } catch {
      scheduleRetry(video);
      return;
    }
    if (!playPromise || typeof playPromise.then !== "function") return;

    pendingPlayback.add(video);
    playPromise
      .then(() => {
        pendingPlayback.delete(video);
        clearRetry(video);
      })
      .catch(() => {
        pendingPlayback.delete(video);
        scheduleRetry(video);
      });
  }

  const syncPlayback = () => {
    const activeVideos = new Set(getPlayableVideos());
    const matrixShouldPlay = (
      matrixInView
      && !document.hidden
      && !reducedMotionQuery.matches
    );

    videos.forEach((video) => {
      const videoShouldPlay = matrixShouldPlay && activeVideos.has(video);
      video.dataset.heroAutoplay = String(videoShouldPlay);

      if (videoShouldPlay) {
        attemptPlayback(video);
      } else {
        pendingPlayback.delete(video);
        clearRetry(video);
        video.autoplay = false;
        video.pause();
      }
    });

    matrix.classList.toggle("is-playing", matrixShouldPlay);
  };

  videos.forEach((video) => {
    video.addEventListener("loadeddata", () => attemptPlayback(video));
    video.addEventListener("canplay", () => attemptPlayback(video));
    video.addEventListener("playing", () => clearRetry(video));
    video.addEventListener("pause", () => scheduleRetry(video));
  });

  if ("IntersectionObserver" in window) {
    const matrixObserver = new IntersectionObserver(
      ([entry]) => {
        matrixInView = entry.isIntersecting;
        syncPlayback();
      },
      { threshold: 0.08 }
    );
    matrixObserver.observe(matrix);
  } else {
    matrixInView = true;
    syncPlayback();
  }

  document.addEventListener("visibilitychange", syncPlayback);
  window.addEventListener("pageshow", syncPlayback);
  window.addEventListener("load", syncPlayback);

  if (typeof mobileHeroQuery.addEventListener === "function") {
    mobileHeroQuery.addEventListener("change", syncPlayback);
    reducedMotionQuery.addEventListener("change", syncPlayback);
  } else {
    mobileHeroQuery.addListener(syncPlayback);
    reducedMotionQuery.addListener(syncPlayback);
  }
}

function setupVideoReels() {
  const videos = Array.from(document.querySelectorAll("[data-reel-video]"));
  if (!videos.length) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const updateState = (video) => {
    const tile = video.closest(".video-tile, .chain-media, .sim-media, .ov-player");
    if (video.closest("[data-view-pane]")?.hidden) return;
    const state = tile?.querySelector(".video-tile-state");
    const isPlaying = !video.paused && !video.ended;
    tile?.classList.toggle("is-playing", isPlaying);
    if (state) state.textContent = isPlaying ? "Playing" : "Paused";
  };

  const toggleVideo = (video) => {
    if (video.paused) {
      video.dataset.userPaused = "false";
      video.autoplay = true;
      loadDeferredVideo(video);
      video.play().catch(() => {});
    } else {
      video.dataset.userPaused = "true";
      video.autoplay = false;
      video.pause();
    }
  };

  videos.forEach((video) => {
    video.muted = true;
    video.playsInline = true;
    video.addEventListener("play", () => updateState(video));
    video.addEventListener("pause", () => updateState(video));
    video.addEventListener("canplay", () => {
      if (video.dataset.inView === "true" && video.dataset.userPaused !== "true") {
        video.play().catch(() => {});
      }
    });
    video.addEventListener("click", () => toggleVideo(video));
    video.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      toggleVideo(video);
    });
  });

  if (reducedMotion) return;

  if ("IntersectionObserver" in window) {
    const videoObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const video = entry.target;
          video.dataset.inView = String(entry.isIntersecting);
          if (entry.isIntersecting && video.dataset.userPaused !== "true") {
            video.autoplay = true;
            loadDeferredVideo(video);
            video.play().catch(() => {});
          } else {
            video.autoplay = false;
            video.pause();
          }
        });
      },
      { threshold: 0.15, rootMargin: "40px 0px 80px" }
    );
    videos.forEach((video) => videoObserver.observe(video));
  } else {
    videos.slice(0, 3).forEach((video) => {
      loadDeferredVideo(video);
      video.play().catch(() => {});
    });
  }
}

function setupOverviewChapters() {
  const video = document.querySelector("[data-ov-video]");
  const buttons = Array.from(document.querySelectorAll("[data-ov-seek]"));
  if (!video || !buttons.length) return;
  const sync = () => {
    const t = video.currentTime || 0;
    buttons.forEach((b) => {
      const a = Number(b.dataset.ovSeek), e = Number(b.dataset.ovEnd);
      const p = Math.max(0, Math.min(1, (t - a) / (e - a)));
      b.classList.toggle("is-active", t >= a && t < e);
      b.style.setProperty("--ov-progress", `${(t >= a ? p : 0) * 100}%`);
    });
  };
  // smooth the fill: drive it with requestAnimationFrame while playing (timeupdate only fires ~4 Hz)
  let raf = 0;
  const tick = () => { sync(); raf = (video.paused || video.ended) ? 0 : requestAnimationFrame(tick); };
  const start = () => { if (!raf) raf = requestAnimationFrame(tick); };
  const stop = () => { if (raf) cancelAnimationFrame(raf); raf = 0; sync(); };
  video.addEventListener("play", start);
  video.addEventListener("playing", start);
  video.addEventListener("pause", stop);
  video.addEventListener("ended", stop);
  video.addEventListener("timeupdate", sync);   // keep synced while paused/scrubbing
  buttons.forEach((b) =>
    b.addEventListener("click", () => {
      loadDeferredVideo(video);
      const seek = () => { video.currentTime = Number(b.dataset.ovSeek); sync(); };
      if (video.readyState >= 1) seek(); else video.addEventListener("loadedmetadata", seek, { once: true });
      video.dataset.userPaused = "false";
      video.play().catch(() => {});
    }),
  );
  sync();
}

function setupSiteChrome() {
  const header = document.getElementById("site-header");
  const hero = document.getElementById("top");
  const stage = hero?.querySelector(".hero-stage");
  const wall = document.getElementById("hero-wall");
  const toggle = document.getElementById("playback-toggle");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // hero wall: portrait or landscape source, fades in once it can play
  if (wall) {
    const pick = () => (window.matchMedia("(max-aspect-ratio: 1/1)").matches ? wall.dataset.portrait : wall.dataset.landscape);
    wall.src = pick();
    wall.addEventListener("canplay", () => wall.classList.add("is-ready"), { once: true });
    if (!reduced) wall.play().catch(() => {});
  }
  // staggered reveals inside grids
  document.querySelectorAll(".contribution-grid, .video-reels, .sim-grid, .result-summary, .paper-grid").forEach((grid) => {
    Array.from(grid.children).forEach((child, i) => {
      child.classList.add("reveal");
      child.style.setProperty("--reveal-delay", `${Math.min(i, 6) * 0.07}s`);
    });
  });
  const onScroll = () => {
    const y = window.scrollY;
    const heroH = hero ? hero.offsetHeight : 0;
    const vh = window.innerHeight;
    // hero dissolve: copy rises and fades while a white veil rises inside the sticky stage, so the
    // stage is already page-white at the bottom when it scrolls away (no seam with the next section)
    const p = Math.max(0, Math.min(1, y / Math.max(1, heroH - vh)));
    if (stage && !reduced) {
      const ease = (u) => u * u * (3 - 2 * u);
      stage.style.setProperty("--hero-copy", String(Math.max(0, 1 - 1.15 * p)));
      stage.style.setProperty("--hero-shift", `${-p * 70}px`);
      stage.style.setProperty("--hero-bright", String(0.72 - 0.2 * p));
      stage.style.setProperty("--hero-scale", String(1 + 0.05 * p));
      stage.style.setProperty("--hero-out", String(ease(p)));
    } else if (stage) {
      stage.style.setProperty("--hero-out", "0");
    }
    if (header) {
      const inHero = y < heroH - Math.max(70, 0.4 * vh);   // light once the header is over the veil
      header.classList.toggle("is-scrolled", inHero && y > 24);
      header.classList.toggle("is-light", !inHero);
    }
    if (toggle) toggle.classList.toggle("is-visible", y > heroH * 0.6);
    if (wall && !reduced) {
      if (y > heroH + 200 && !wall.paused) wall.pause();
      else if (y <= heroH && wall.paused && !document.body.classList.contains("videos-paused")) wall.play().catch(() => {});
    }
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);
  onScroll();
  // one switch for every video on the page
  if (toggle) {
    toggle.addEventListener("click", () => {
      const paused = !document.body.classList.contains("videos-paused");
      document.body.classList.toggle("videos-paused", paused);
      toggle.setAttribute("aria-pressed", String(paused));
      toggle.textContent = paused ? "Play videos" : "Pause videos";
      document.querySelectorAll("video").forEach((v) => {
        if (paused) { v.dataset.userPaused = "true"; v.pause(); }
        else {
          v.dataset.userPaused = "false";
          const r = v.getBoundingClientRect();
          if (r.bottom > 0 && r.top < window.innerHeight && v.offsetParent !== null) v.play().catch(() => {});
        }
      });
    });
  }
}

function setupViewSwitches() {
  document.querySelectorAll(".sim-wide").forEach((fig) => {
    const buttons = Array.from(fig.querySelectorAll("[data-view]"));
    if (!buttons.length) return;
    const panes = Array.from(fig.querySelectorAll("[data-view-pane]"));
    buttons.forEach((b) => b.addEventListener("click", () => {
      const cur = panes.find((p) => !p.hidden)?.querySelector("video");
      const t = cur ? cur.currentTime : 0;
      buttons.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      panes.forEach((p) => {
        const on = p.dataset.viewPane === b.dataset.view; p.hidden = !on;
        const v = p.querySelector("video");
        if (!v) return;
        if (on) { loadDeferredVideo(v); const go = () => { v.currentTime = t; v.play().catch(() => {}); }; v.readyState >= 1 ? go() : v.addEventListener("loadedmetadata", go, { once: true }); }
        else v.pause();
      });
    }));
  });
}

function setupSimTabs() {
  const tabs = Array.from(document.querySelectorAll("[data-sim-tab]"));
  if (!tabs.length) return;
  const select = (tab, focus) => {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      const panel = document.querySelector(`[data-sim-panel="${t.dataset.simTab}"]`);
      if (!panel) return;
      panel.hidden = !on;
      if (!on) panel.querySelectorAll("video").forEach((v) => v.pause());
    });
    if (focus) tab.focus();
  };
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (event) => {
      const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
      if (!step) return;
      event.preventDefault();
      select(tabs[(i + step + tabs.length) % tabs.length], true);
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupSiteChrome();
  setupSimTabs();
  setupViewSwitches();
  setupOverviewChapters();
  const topButton = document.querySelector(".scroll-to-top");
  const navLinks = Array.from(document.querySelectorAll(".outline-link"));
  const sections = navLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  const updateNavigation = () => {
    if (topButton) {
      topButton.classList.toggle("is-visible", window.scrollY > 420);
    }

    // sections are reordered visually with CSS `order`, so pick by on-screen position
    let activeId = sections[0]?.id, best = -Infinity;
    for (const section of sections) {
      const top = section.getBoundingClientRect().top;
      if (top <= 180 && top > best) { best = top; activeId = section.id; }
    }
    navLinks.forEach((link) => {
      link.classList.toggle("is-active", link.getAttribute("href") === `#${activeId}`);
    });
  };

  const revealElements = Array.from(document.querySelectorAll(".reveal"));
  const revealVisibleElements = () => {
    revealElements.forEach((element) => {
      const rect = element.getBoundingClientRect();
      if (rect.top < window.innerHeight * 1.08 && rect.bottom > -80) {
        element.classList.add("is-visible");
      }
    });
  };

  if ("IntersectionObserver" in window) {
    const revealObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            revealObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.08, rootMargin: "80px 0px" }
    );
    revealElements.forEach((element) => revealObserver.observe(element));
  } else {
    revealElements.forEach((element) => element.classList.add("is-visible"));
  }

  revealVisibleElements();
  setupHeroVideoMatrix();
  setupVideoReels();
  setupMathFormula();
  setupGammaChart();
  window.setTimeout(revealVisibleElements, 120);
  window.addEventListener("scroll", updateNavigation, { passive: true });
  window.addEventListener("load", () => {
    if (!window.location.hash) return;
    const target = document.querySelector(window.location.hash);
    if (!target) return;
    window.setTimeout(() => {
      target.scrollIntoView();
      revealVisibleElements();
      updateNavigation();
    }, 80);
  });
  updateNavigation();
});

document.addEventListener('DOMContentLoaded', () => {
  const launch = document.getElementById('launch-fiber-playground');
  if (!launch) return;
  launch.addEventListener('click', () => {
    document.getElementById('fiber-playground').src = 'playground/?autostart=1';
    document.getElementById('fiber-playground-cover').hidden = true;
  }, { once: true });
});
