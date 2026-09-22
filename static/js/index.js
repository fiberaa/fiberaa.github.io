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
    const tile = video.closest(".video-tile, .chain-media");
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

document.addEventListener("DOMContentLoaded", () => {
  const topButton = document.querySelector(".scroll-to-top");
  const navLinks = Array.from(document.querySelectorAll(".outline-link"));
  const sections = navLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  const updateNavigation = () => {
    if (topButton) {
      topButton.classList.toggle("is-visible", window.scrollY > 420);
    }

    let activeId = sections[0]?.id;
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= 180) activeId = section.id;
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
