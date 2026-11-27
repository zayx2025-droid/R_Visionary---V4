// The model and inference run inside this worker, not in the page's UI thread.
let generator;
self.onmessage=async ({data:m})=>{
 const {id,kind}=m;
 try{
  if(kind==='load'){
   const {pipeline,env}=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1');
   env.allowLocalModels=false;
   env.backends.onnx.wasm.numThreads=1;
   generator=await pipeline('text-generation','onnx-community/SmolLM2-360M-Instruct-ONNX',{device:'wasm',dtype:'q4',progress_callback:p=>self.postMessage({id,progress:p.status==='progress'?`${p.file||'Model'}: ${Math.round(p.progress||0)}%`:String(p.status||'Preparing model')})});
   self.postMessage({id,result:true});
  }else if(kind==='prompt'){
   if(!generator)throw Error('CPU model not loaded');
   const out=await generator([{role:'system',content:m.system},{role:'user',content:m.prompt}],{max_new_tokens:m.tokens,do_sample:false,return_full_text:false});
   const value=out[0]?.generated_text;
   const text=typeof value==='string'?value:Array.isArray(value)?value.filter(x=>x.role==='assistant').at(-1)?.content:'';
   if(!text)throw Error('CPU model returned no text');
   self.postMessage({id,result:text});
  }else throw Error('Unknown worker action');
 }catch(e){self.postMessage({id,error:String(e?.message||e)});}
};
