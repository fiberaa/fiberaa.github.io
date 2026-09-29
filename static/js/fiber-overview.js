/* Schematic FIBER illustration. No recorded policy outputs or measured losses. */
(() => {
  const root=document.querySelector('.fiber-overview');if(!root)return;
  const world=root.querySelector('[data-fiber-world]'), play=root.querySelector('[data-fiber-play]'), slider=root.querySelector('[data-fiber-time]');
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const mobile=matchMedia('(max-width:600px)');
  const ink='#344c5a', muted='#87938f', teal='#567e79', clay='#b47d5e', pale='#e4e8e3';
  const titles=['Heterogeneous skill acquisition','Action-aligned latent policy projection','Closed-loop latent rollouts','Offline planner training','Unified latent flow planning','Execution through the frozen BFM'];
  const details=[
    'Acquire interaction experts independently through motion imitation, reward shaping, or SF-guided behavior reuse. Each expert learns a physical skill.',
    'At the same robot states, match the expert’s actions to actions induced by a projected latent through the frozen BFM, over a short horizon. Train the projector; keep the BFM frozen.',
    'Execute each projected skill through the frozen BFM. Collect latent trajectories together with depth, proprioception, and steering conditions; the original experts are no longer needed.',
    'Aggregate the collected latent rollouts and train a unified perceptive flow planner entirely offline.',
    'Given onboard depth, proprioception, and a steering token, the planner predicts future action-aligned latents.',
    'The frozen BFM maps the planned latent and current robot state to whole-body actions. State feedback remains active at high frequency.'
  ];
  let time=0,playing=!reduced.matches,visible=false,last=null;
  const clamp=v=>Math.max(0,Math.min(1,v)),ease=v=>{v=clamp(v);return v*v*(3-2*v);},lerp=(a,b,q)=>a+(b-a)*q;
  const label=(s,x,y,size=14,color=ink,anchor='middle')=>`<text x="${x}" y="${y}" font-size="${mobile.matches?size*1.12:size}" fill="${color}" text-anchor="${anchor}">${s}</text>`;
  const path=(pts,color=ink,width=2,extra='')=>`<path d="${pts.map((p,i)=>(i?'L':'M')+p.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${extra}/>`;
  const circle=(x,y,r,color)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`;
  const frame=(x,y,w,h,name,sub='',accent=false)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${accent?'#edf2ee':'#faf9f6'}" stroke="${accent?'#9bb3a9':'#d9ded7'}"/>${label(name,x+w/2,y+h/2+(sub?-5:5),14)}${sub?label(sub,x+w/2,y+h/2+16,11,muted):''}`;
  const arrow=(x1,y1,x2,y2,color=muted,extra='')=>`<path d="M${x1},${y1}L${x2},${y2}" fill="none" stroke="${color}" stroke-width="1.4" marker-end="url(#fiber-arrow)" ${extra}/>`;
  const token=(x,y,i,alpha=1)=>`<g opacity="${alpha}"><rect x="${x}" y="${y}" width="25" height="33" rx="4" fill="#e6eeea" stroke="#a6bcb1"/>${[0,1,2,3].map(j=>`<rect x="${x+6}" y="${y+6+j*5}" width="${7+(i*7+j*3)%12}" height="2" rx="1" fill="${teal}" opacity="${.3+j*.15}"/>`).join('')}</g>`;
  // Images remain mounted across the entire story. Only their transforms change.
  const ns='http://www.w3.org/2000/svg';
  const node=tag=>document.createElementNS(ns,tag);
  const artLayer=node('g'), diagramLayer=node('g');
  world.append(artLayer,diagramLayer);
  const art=new Map();
  ['box','door','grasp'].forEach((skill,index)=>{
    const g=node('g'),w=202,h=w*817/641,id=`fiber-art-${skill}`;
    g.innerHTML=`<defs><clipPath id="${id}"><rect x="-90" y="${-h}" width="${w}" height="${h}"/></clipPath></defs><g clip-path="url(#${id})"><image href="static/images/fiber-illustration/humanoid-skills-v1.png" x="${-90-index*w}" y="${-h}" width="${w*3}" height="${h}" preserveAspectRatio="none"/></g>`;
    g.setAttribute('opacity','0');artLayer.append(g);
    art.set(skill,{g,current:null,target:null});
  });
  let artTargets=new Map(),diagramKey=null,diagram=null,diagramAnimations=[];
  const robot=(x,y,scale,progress,skill='box',color=ink,opacity=1)=>{
    artTargets.set(skill,{x,y,scale,opacity});return '';
  };
  // Update attributes in place; never remove/reload artwork on animation frames.
  const patch=(current,next)=>{
    for(const a of [...current.attributes])if(!next.hasAttribute(a.name))current.removeAttribute(a.name);
    for(const a of next.attributes)if(current.getAttribute(a.name)!==a.value)current.setAttribute(a.name,a.value);
    if(!next.children.length){if(current.textContent!==next.textContent)current.textContent=next.textContent;return;}
    while(current.children.length>next.children.length)current.lastElementChild.remove();
    [...next.children].forEach((child,i)=>{
      const old=current.children[i];
      if(!old)current.append(child.cloneNode(true));
      else if(old.tagName!==child.tagName)old.replaceWith(child.cloneNode(true));
      else patch(old,child);
    });
  };
  const paintDiagram=(markup,key)=>{
    const next=node('g');next.innerHTML=markup;
    // The arrow definition is global to the current diagram; no duplicate IDs.
    if(key===diagramKey){patch(diagram,next);return;}
    diagramAnimations.forEach(a=>a.cancel());diagramAnimations=[];
    [...diagramLayer.children].filter(g=>g!==diagram).forEach(g=>g.remove());
    const previous=diagram;diagram=next;diagramKey=key;diagramLayer.append(next);
    if(reduced.matches||!previous){previous?.remove();return;}
    previous.querySelectorAll('[id]').forEach(el=>el.removeAttribute('id'));
    const out=previous.animate([{opacity:1},{opacity:0}],{duration:300,fill:'forwards',easing:'ease-out'});
    out.finished.then(()=>previous.remove()).catch(()=>{});
    const incoming=next.animate([{opacity:0,transform:'translateY(6px)'},{opacity:1,transform:'translateY(0)'}],{duration:650,delay:100,easing:'cubic-bezier(.22,1,.36,1)',fill:'backwards'});
    diagramAnimations.push(out,incoming);
  };
  const animateArt=dt=>{
    const q=reduced.matches?1:1-Math.exp(-dt/160);
    for(const item of art.values()){
      if(!item.target)continue;
      if(!item.current)item.current={...item.target,opacity:0};
      for(const k of ['x','y','scale','opacity'])item.current[k]=lerp(item.current[k],item.target[k],q);
      const c=item.current;item.g.setAttribute('transform',`translate(${c.x} ${c.y}) scale(${c.scale})`);item.g.setAttribute('opacity',c.opacity);
    }
  };
  const dataset=(x,y,amount=1)=>{
    let s='';for(let row=0;row<3;row++){for(let col=0;col<6;col++)s+=token(x+col*33,y+row*43,row*6+col,amount);}
    return s+label('Collected latent rollouts',x+95,y+151,13,muted);
  };
  const draw=()=>{
    artTargets=new Map();
    const phase=Math.min(5,Math.floor(time)),q=time-phase;
    root.dataset.stage=phase;root.dataset.playing=playing;
    root.querySelector('[data-fiber-phase]').textContent=titles[phase];root.querySelector('[data-fiber-detail]').textContent=details[phase];
    play.textContent=playing?'Pause':'Play';slider.value=Math.round(time/6*1000);
    root.querySelectorAll('[data-fiber-jump]').forEach((b,i)=>b.setAttribute('aria-current',i===phase));
    let s=`<defs><marker id="fiber-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1,1L8,5L1,9" fill="none" stroke="${muted}" stroke-width="1.5"/></marker></defs>`;
    const motion=(1-Math.cos(q*Math.PI*2))*.5;
    // Start with physical skills; the box expert moves into a persistent workspace.
    if(phase===0){
      const converge=ease((q-.72)/.28);
      s+=robot(lerp(155,195,converge),285,.88,motion,'box');
      s+=robot(450,285,.88,motion,'door',ink,1-converge);
      s+=robot(735,285,.88,motion,'grasp',ink,1-converge);
      s+=label('Box pickup',155,326,15)+`<g opacity="${1-converge}">${label('Door opening',450,326,15)}${label('Object grasping',735,326,15)}</g>`;

    }else if(phase===1){
      // Freeze the physical state while two policies produce action responses.
      s+=robot(195,284,.94,.38,'box');s+=label('Same robot state sₜ',195,327,14,muted);
      s+=frame(350,50,132,48,'Expert policy','πᴱ');s+=frame(350,161,132,48,'Projector','ρφ(o) → z',true);
      s+=frame(550,161,135,48,'Frozen BFM','πBFM(s, z)');
      s+=arrow(290,144,330,74)+arrow(290,144,330,185)+arrow(482,185,542,185);
      s+=token(500,168,1);
      s+=label('Expert actions aᴱ',758,44,12,clay)+label('BFM actions a',758,155,12,teal);
      const fit=ease(q/.72);
      for(let j=0;j<5;j++){
        const target=[.72,.37,.9,.48,.62][j],value=lerp([.31,.8,.49,.78,.28][j],target,fit);
        s+=`<rect x="${716+j*18}" y="${109-target*51}" width="10" height="${target*51}" rx="2" fill="${clay}"/>`;
        s+=`<rect x="${716+j*18}" y="${220-value*51}" width="10" height="${value*51}" rx="2" fill="${teal}"/>`;
        s+=path([[721+j*18,116],[721+j*18,153]],'#d7dcd5',1,'stroke-dasharray="2 4"');
      }
      s+=arrow(488,74,692,74,clay)+arrow(685,185,701,185,teal);
      s+=label('Match actions over a short horizon',611,271,15);
      s+=label('Lact = mean ‖πBFM(s, z) − aᴱ‖²',611,298,16,teal);
      s+=label('Semantic regularization: z ≈ B(s)',611,326,12,muted);
      s+=label('Illustrative action components',758,241,10,muted);
      s+=`<path d="M820,250L842,250L842,350L416,350L416,217" fill="none" stroke="${teal}" stroke-width="1.3" marker-end="url(#fiber-arrow)" opacity=".45"/>`;
    }else if(phase===2){
      s+=robot(195,284,.94,motion,'box')+label('Projected skill execution',195,327,14,muted);
      s+=frame(350,123,145,60,'Frozen BFM','Current state + latent');
      s+=arrow(342,153,302,153)+arrow(500,153,565,153);
      s+=dataset(600,95);
      for(let i=0;i<3;i++){const u=(q*2+i/3)%1;s+=token(lerp(512,575,u),138,i,Math.sin(u*Math.PI));}
      s+=label('Depth · proprioception · steering conditions',620,304,13,muted);
    }else if(phase===3){
      s+=dataset(105,106)+arrow(312,167,405,167);
      s+=frame(416,121,190,92,'DiT flow planner','Offline training',true);
      s+=arrow(618,167,690,167);
      for(let i=0;i<4;i++)s+=token(704+i*32,150,i);
      s+=label('Future latent trajectory',766,227,13,muted);
      s+=label('Latent rollouts + observation and steering conditions',356,310,14,muted);
      const u=(q*2)%1;s+=token(lerp(326,375,u),151,2,Math.sin(u*Math.PI));
    }else{
      s+=robot(720,284,.94,motion,'box')+label('Whole-body interaction',735,327,14,muted);
      s+=label('Onboard depth',138,87,13)+label('Proprioception',138,115,13)+label('Steering token',138,143,13);
      s+=arrow(207,118,247,118);
      s+=frame(256,81,152,76,'Flow planner','Lower-rate planning',true);
      s+=arrow(412,119,439,119);
      for(let i=0;i<4;i++)s+=token(447+i*32,101,i,phase===4?.55+.45*Math.sin(q*5+i)**2:1);
      s+=label('Action-aligned latents',510,78,12,teal)+label('zₖ, zₖ₊₁, …',510,157,13,teal);
      s+=frame(437,217,145,60,'Frozen BFM','πBFM(sₜ, zₖ)');
      s+=arrow(510,168,510,209)+arrow(589,247,652,247);
      s+=path([[732,296],[732,350],[510,350],[510,285]],muted,1.2,'marker-end="url(#fiber-arrow)"');
      s+=label('High-frequency state feedback',619,372,12,muted);
      if(phase===5){const u=q*3%1;s+=circle(lerp(590,649,u),247,3,teal);}
      s+=label(phase===4?'Plan future latents':'Execute through the frozen BFM',315,42,15,muted);
    }
    if(mobile.matches && [1,2,4,5].includes(phase)) {
      const cut=phase===1?s.indexOf('<rect x="350" y="50"'):phase===2?s.indexOf('<rect x="350" y="123"'):s.indexOf('<text x="138"');
      const defsEnd=s.indexOf('</defs>')+7;
      let first=s.slice(defsEnd,cut);
      if(phase===1)first=first.replace(arrow(290,144,330,74),'').replace(arrow(290,144,330,185),'');
      let second=s.slice(cut);
      if(phase===1)second=second.replace(arrow(290,144,330,74),'').replace(arrow(290,144,330,185),'');
      const transform=phase===1?'translate(-295 195) scale(.9)':phase===2?'translate(-270 195) scale(.87)':'translate(-30 205) scale(.65)';
      const robotTransform=phase<3?'translate(98 -8) scale(.65)':'translate(-245 -8) scale(.65)';
      s=s.slice(0,defsEnd)+`<g transform="${robotTransform}">${first}</g><g transform="${transform}">${second}</g>`;
      world.setAttribute('viewBox','0 0 480 540');
    } else if(mobile.matches){
      const defsEnd=s.indexOf('</defs>')+7;
      s=s.slice(0,defsEnd)+`<g transform="translate(0 130) scale(.53)">${s.slice(defsEnd)}</g>`;
      world.setAttribute('viewBox','0 0 480 540');
    } else world.setAttribute('viewBox','0 0 900 380');
    for(const [skill,item] of art){
      let target=artTargets.get(skill);
      if(target&&mobile.matches&&[1,2,4,5].includes(phase))target={...target,x:target.x*.65+(phase<3?98:-245),y:target.y*.65-8,scale:target.scale*.65};
      if(target&&mobile.matches&&![1,2,4,5].includes(phase))target={...target,x:target.x*.53,y:target.y*.53+130,scale:target.scale*.53};
      item.target=target||(item.target?{...item.target,opacity:0}:null);
    }
    paintDiagram(s,`${phase}-${mobile.matches}`);
    if(reduced.matches)animateArt(1000);
  };
  play.onclick=()=>{playing=!playing;last=null;draw();};
  root.querySelector('[data-fiber-replay]').onclick=()=>{time=0;playing=!reduced.matches;last=null;draw();};
  slider.oninput=()=>{time=Math.min(5.999,Number(slider.value)/1000*6);playing=false;last=null;draw();};
  root.querySelectorAll('[data-fiber-jump]').forEach(b=>b.onclick=()=>{time=Number(b.dataset.fiberJump)+.15;playing=false;last=null;draw();});
  reduced.addEventListener('change',e=>{if(e.matches){playing=false;last=null;diagramAnimations.forEach(a=>a.cancel());diagramAnimations=[];[...diagramLayer.children].filter(g=>g!==diagram).forEach(g=>g.remove());draw();}});
  new IntersectionObserver(e=>{visible=e[0].isIntersecting;last=null;},{threshold:.1}).observe(root);
  let lastArt=null,lastDraw=0;
  const tick=now=>{
    if(visible&&!document.hidden){
      const dt=lastArt===null?16:Math.min(now-lastArt,100);lastArt=now;
      if(playing){if(last!==null)time=(time+Math.min(now-last,100)/8500)%6;last=now;if(now-lastDraw>32){draw();lastDraw=now;}}else last=null;
      animateArt(dt);
    }else{last=null;lastArt=null;}
    requestAnimationFrame(tick);
  };
  mobile.addEventListener('change',draw);
  draw();requestAnimationFrame(tick);
})();
