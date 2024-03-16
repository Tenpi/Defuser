import React, {useContext, useEffect, useState, useRef, createElement} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, StepsContext, AmountContext,
CFGContext, SizeContext, DenoiseContext, SamplerContext, SeedContext, InterrogateTextContext, PromptContext, NegativePromptContext,
ClipSkipContext, ModelNameContext, RenderImageContext, VAENameContext, InterrogatorNameContext, ProcessingContext, ImageInputContext,
FormatContext, DeletionContext, TextualInversionsContext, HypernetworksContext, LorasContext, MaskImageContext, PrecisionContext,
ControlImageContext, ControlProcessorContext, ControlScaleContext, ControlGuessModeContext, ControlStartContext, ControlEndContext,
ControlInvertContext, StyleFidelityContext, ControlReferenceImageContext, HorizontalExpandContext, VerticalExpandContext, UpscalerContext,
ExpandImageContext, ExpandMaskContext, StartedContext, SocketContext, LoopModeContext, SavedPromptsContext, WatermarkContext, NSFWTabContext,
InvisibleWatermarkContext, SauceNaoAPIKeyContext, RandomPromptModeContext, GeneratorContext, NovelAITokenContext, HolaraAICookieContext,
SavedPromptsNovelAIContext, SavedPromptsHolaraAIContext, ModelDirContext, OutputDirContext, XAdaptModelContext, FreeUContext, IPAdapterContext,
IPProcessorContext, IPImageContext, IPWeightContext, IPMaskImageContext, FramesContext, ThemeContext, ThemeSelectorContext} from "../Context"
import functions from "../structures/Functions"
import checkbox from "../assets/icons/checkbox2.png"
import checkboxChecked from "../assets/icons/checkbox2-checked.png"
import loop from "../assets/icons/loop.png"
import upscale from "../assets/icons/upscale.png"
import appendToPromptIcon from "../assets/icons/append-to-prompt.png"
import diceIcon from "../assets/icons/dice.png"
import bookmarkIcon from "../assets/icons/bookmark.png"
import axios from "axios"
import "./styles/generatebar.less"

const GenerateBar: React.FunctionComponent = (props) => {
    const {theme, setTheme} = useContext(ThemeContext)
    const {themeSelector, setThemeSelector} = useContext(ThemeSelectorContext)
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {steps, setSteps} = useContext(StepsContext)
    const {cfg, setCFG} = useContext(CFGContext)
    const {frames, setFrames} = useContext(FramesContext)
    const {size, setSize} = useContext(SizeContext)
    const {denoise, setDenoise} = useContext(DenoiseContext)
    const {seed, setSeed} = useContext(SeedContext)
    const {sampler, setSampler} = useContext(SamplerContext)
    const {prompt, setPrompt} = useContext(PromptContext)
    const {negativePrompt, setNegativePrompt} = useContext(NegativePromptContext)
    const {clipSkip, setClipSkip} = useContext(ClipSkipContext)
    const {interrogateText, setInterrogateText} = useContext(InterrogateTextContext)
    const {interrogatorName, setInterrogatorName} = useContext(InterrogatorNameContext)
    const {upscaler, setUpscaler} = useContext(UpscalerContext)
    const {modelName, setModelName} = useContext(ModelNameContext)
    const {vaeName, setVAEName} = useContext(VAENameContext)
    const {processing, setProcessing} = useContext(ProcessingContext)
    const {maskImage, setMaskImage} = useContext(MaskImageContext)
    const {ipMaskImage, setIPMaskImage} = useContext(IPMaskImageContext)
    const [infinite, setInfinite] = useState(false)
    const [upscaling, setUpscaling] = useState(true)
    const [start, setStart] = useState(false)
    const {amount, setAmount} = useContext(AmountContext)
    const {renderImage, setRenderImage} = useContext(RenderImageContext)
    const {imageInput, setImageInput} = useContext(ImageInputContext)
    const {deletion, setDeletion} = useContext(DeletionContext)
    const {format, setFormat} = useContext(FormatContext)
    const {precision, setPrecision} = useContext(PrecisionContext)
    const {textualInversions, setTextualInversions} = useContext(TextualInversionsContext)
    const {hypernetworks, setHypernetworks} = useContext(HypernetworksContext)
    const {loras, setLoras} = useContext(LorasContext)
    const {controlImage, setControlImage} = useContext(ControlImageContext)
    const {controlProcessor, setControlProcessor} = useContext(ControlProcessorContext)
    const {controlScale, setControlScale} = useContext(ControlScaleContext)
    const {controlGuessMode, setControlGuessMode} = useContext(ControlGuessModeContext)
    const {controlStart, setControlStart} = useContext(ControlStartContext)
    const {controlReferenceImage, setControlReferenceImage} = useContext(ControlReferenceImageContext)
    const {controlEnd, setControlEnd} = useContext(ControlEndContext)
    const {controlInvert, setControlInvert} = useContext(ControlInvertContext)
    const {styleFidelity, setStyleFidity} = useContext(StyleFidelityContext)
    const {expandImage, setExpandImage} = useContext(ExpandImageContext)
    const {expandMask, setExpandMask} = useContext(ExpandMaskContext)
    const {horizontalExpand, setHorizontalExpand} = useContext(HorizontalExpandContext)
    const {verticalExpand, setVerticalExpand} = useContext(VerticalExpandContext)
    const {started, setStarted} = useContext(StartedContext)
    const [interrogateStarted, setInterrogateStarted] = useState(false)
    const {socket, setSocket} = useContext(SocketContext)
    const {loopMode, setLoopMode} = useContext(LoopModeContext)
    const {savedPrompts, setSavedPrompts} = useContext(SavedPromptsContext)
    const {watermark, setWatermark} = useContext(WatermarkContext)
    const {invisibleWatermark, setInvisibleWatermark} = useContext(InvisibleWatermarkContext)
    const {nsfwTab, setNSFWTab} = useContext(NSFWTabContext)
    const {saucenaoAPIKey, setSaucenaoAPIKey} = useContext(SauceNaoAPIKeyContext)
    const {randomPromptMode, setRandomPromptMode} = useContext(RandomPromptModeContext)
    const {generator, setGenerator} = useContext(GeneratorContext)
    const {novelAIToken, setNovelAIToken} = useContext(NovelAITokenContext)
    const {holaraAICookie, setHolaraAICookie} = useContext(HolaraAICookieContext)
    const {savedPromptsNovelAI, setSavedPromptsNovelAI} = useContext(SavedPromptsNovelAIContext)
    const {savedPromptsHolaraAI, setSavedPromptsHolaraAI} = useContext(SavedPromptsHolaraAIContext)
    const {xAdaptModel, setXAdaptModel} = useContext(XAdaptModelContext)
    const {freeU, setFreeU} = useContext(FreeUContext)
    const {modelDir, setModelDir} = useContext(ModelDirContext)
    const {outputDir, setOutputDir} = useContext(OutputDirContext)
    const {ipAdapter, setIPAdapter} = useContext(IPAdapterContext)
    const {ipProcessor, setIPProcessor} = useContext(IPProcessorContext)
    const {ipWeight, setIPWeight} = useContext(IPWeightContext)
    const {ipImage, setIPImage} = useContext(IPImageContext)
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        let saturation = siteSaturation
        let lightness = siteLightness
        if (themeSelector === "original") {
            if (theme === "light") saturation -= 60
            if (theme === "light") lightness += 90
        } else if (themeSelector === "accessibility") {
            if (theme === "light") saturation -= 90
            if (theme === "light") lightness += 200
            if (theme === "dark") saturation -= 50
            if (theme === "dark") lightness -= 30
        }
        return `hue-rotate(${siteHue - 180}deg) saturate(${saturation}%) brightness(${lightness + 50}%)`
    }

    useEffect(() => {
        const savedPrompt = localStorage.getItem("prompt")
        if (savedPrompt) setPrompt(savedPrompt)
        const savedNegativePrompt = localStorage.getItem("negativePrompt")
        if (savedNegativePrompt) setNegativePrompt(savedNegativePrompt)
        const savedInterrogatorName = localStorage.getItem("interrogatorName")
        if (savedInterrogatorName) setInterrogatorName(savedInterrogatorName)
        const savedProcessing = localStorage.getItem("processing")
        if (savedProcessing) setProcessing(savedProcessing)
        const savedInfinite = localStorage.getItem("infinite")
        if (savedInfinite) setInfinite(savedInfinite === "true")
        const savedUpscaling = localStorage.getItem("upscaling")
        if (savedUpscaling) setUpscaling(savedUpscaling === "true")
        const savedFormat = localStorage.getItem("format")
        if (savedFormat) setFormat(savedFormat)
        const savedDeletion = localStorage.getItem("deletion")
        if (savedDeletion) setDeletion(savedDeletion)
        const savedPrecision = localStorage.getItem("precision")
        if (savedPrecision) setPrecision(savedPrecision)
        const savedLoopMode = localStorage.getItem("loopMode")
        if (savedLoopMode) setLoopMode(savedLoopMode)
        const savedWatermark = localStorage.getItem("watermark")
        if (savedWatermark) setWatermark(savedWatermark === "true")
        const savedNSFWTab = localStorage.getItem("nsfwTab")
        if (savedNSFWTab) setNSFWTab(savedNSFWTab === "true")
        const savedInvisibleWatermark = localStorage.getItem("invisibleWatermark")
        if (savedInvisibleWatermark) setInvisibleWatermark(savedInvisibleWatermark === "true")
        const savedSaucenaoAPIKey = localStorage.getItem("saucenaoAPIKey")
        if (savedSaucenaoAPIKey) setSaucenaoAPIKey(savedSaucenaoAPIKey)
        const savedRandomPromptMode = localStorage.getItem("randomPromptMode")
        if (savedRandomPromptMode) setRandomPromptMode(savedRandomPromptMode)
        const savedGenerator = localStorage.getItem("generator")
        if (savedGenerator) setGenerator(savedGenerator)
        const savedNovelAIToken = localStorage.getItem("novelAIToken")
        if (savedNovelAIToken) setNovelAIToken(savedNovelAIToken)
        const savedHolaraAICookie = localStorage.getItem("holaraAICookie")
        if (savedHolaraAICookie) setHolaraAICookie(savedHolaraAICookie)
        const savedPrompts = localStorage.getItem("savedPrompts")
        if (savedPrompts) setSavedPrompts(JSON.parse(savedPrompts))
        const savedPromptsNovelAI = localStorage.getItem("savedPrompts-novel-ai")
        if (savedPromptsNovelAI) setSavedPromptsNovelAI(JSON.parse(savedPromptsNovelAI))
        const savedPromptsHolaraAI = localStorage.getItem("savedPrompts-holara-ai")
        if (savedPromptsHolaraAI) setSavedPromptsHolaraAI(JSON.parse(savedPromptsHolaraAI))
        const savedModelDir = localStorage.getItem("modelDir")
        if (savedModelDir) setModelDir(savedModelDir)
        const savedOutputDir = localStorage.getItem("outputDir")
        if (savedOutputDir) setOutputDir(savedOutputDir)
    }, [])

    useEffect(() => {
        if (prompt) localStorage.setItem("prompt", String(prompt))
        if (negativePrompt) localStorage.setItem("negativePrompt", String(negativePrompt))
        localStorage.setItem("infinite", String(infinite))
        localStorage.setItem("upscaling", String(upscaling))
    }, [prompt, negativePrompt, infinite, upscaling])

    const searchDirectory = (files: any[], name: string) => {
        for (let i = 0; i < files.length; i++) {
            if (files[i].directory) {
                const found = searchDirectory(files[i].files, name)
                if (found) return found
            } else {
                if (name === files[i].name) {
                    return files[i].model
                }
            }
        }
        return null
    }

    const getActiveTextualInversions = () => {
        const promptArr = prompt.split(",")
        const negativePromptArr = negativePrompt.split(",")
        const combinedArr = [...promptArr, ...negativePromptArr]
        let active = [] as any
        for (let i = 0; i < combinedArr.length; i++) {
            const name = combinedArr[i].trim().toLowerCase()
            const model = searchDirectory(textualInversions, name)
            if (model) active.push({name, model})
        }
        return active
    }

    const getActiveHypernetworks = () => {
        const promptHypernets = prompt.match(/(<hypernet).*?(>)/gi) || []
        const negativePromptHypernets = negativePrompt.match(/(<hypernet).*?(>)/gi) || []
        const activeHypernets = [...promptHypernets, ...negativePromptHypernets]
        let active = [] as any
        for (let i = 0; i < activeHypernets.length; i++) {
            const hypernetArr = activeHypernets[i].split(":")
            const name = hypernetArr[1]
            const weight = hypernetArr[2].replace(">", "")
            const model = searchDirectory(hypernetworks, name)
            active.push({name, weight, model})
        }
        return active
    }

    const getActiveLoRAs = () => {
        const promptLoras = prompt.match(/(<lora).*?(>)/gi) || []
        const negativePromptLoras = negativePrompt.match(/(<lora).*?(>)/gi) || []
        const activeLoras = [...promptLoras, ...negativePromptLoras]
        let active = [] as any
        for (let i = 0; i < activeLoras.length; i++) {
            const loraArr = activeLoras[i].split(":")
            const name = loraArr[1]
            const weight = loraArr[2].replace(">", "")
            const model = searchDirectory(loras, name)
            active.push({name, weight, model})
        }
        return active
    }

    const getActiveMask = async () => {
        if (!maskImage) return expandMask ? expandMask : null
        const maskImg = document.createElement("img")
        await new Promise<void>((resolve) => {
            maskImg.onload = () => resolve()
            maskImg.src = maskImage
        })
        const normalized = functions.getNormalizedDimensions(maskImg)
        const maskCanvas = document.createElement("canvas")
        maskCanvas.width = normalized.width
        maskCanvas.height = normalized.height 
        const maskCtx = maskCanvas.getContext("2d")!
        maskCtx.drawImage(maskImg, 0, 0, maskCanvas.width, maskCanvas.height)
        const maskPixels = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data
        let active = false
        for (let i = 0; i < maskPixels.length; i+=4) {
            if (maskPixels[i] < 10 && maskPixels[i+1] < 10 && maskPixels[i+2] < 10) {
                // ignore
            } else {
                active = true 
                break
            }
        }
        if (active) {
            if (expandMask) {
                const combined = await functions.combineMasks(expandMask, await functions.expandMask(maskImage, horizontalExpand, verticalExpand))
                return combined ? combined : maskImage
            } else {
                return maskImage
            }
        }
        return null
    }

    const getActiveIPMask = async () => {
        if (!ipMaskImage) return null
        const maskImg = document.createElement("img")
        await new Promise<void>((resolve) => {
            maskImg.onload = () => resolve()
            maskImg.src = ipMaskImage
        })
        const normalized = functions.getNormalizedDimensions(maskImg)
        const maskCanvas = document.createElement("canvas")
        maskCanvas.width = normalized.width
        maskCanvas.height = normalized.height 
        const maskCtx = maskCanvas.getContext("2d")!
        maskCtx.drawImage(maskImg, 0, 0, maskCanvas.width, maskCanvas.height)
        const maskPixels = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data
        let active = false
        for (let i = 0; i < maskPixels.length; i+=4) {
            if (maskPixels[i] < 10 && maskPixels[i+1] < 10 && maskPixels[i+2] < 10) {
                // ignore
            } else {
                active = true 
                break
            }
        }
        if (active) {
            return maskImage
        }
        return null
    }

    const getNormalizedImage = async () => {
        const img = document.createElement("img")
        await new Promise<void>((resolve) => {
            img.onload = () => resolve()
            img.src = expandImage ? expandImage : imageInput
        })
        const normalized = functions.getNormalizedDimensions(img)
        const canvas = document.createElement("canvas")
        canvas.width = normalized.width
        canvas.height = normalized.height 
        const ctx = canvas.getContext("2d")!
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        return canvas.toDataURL("image/png")
    }

    const getPrompt = async (looping?: boolean) => {
        if (looping) {
            if (loopMode === "random prompt") {
                const randomPrompt = await randomizePrompt()
                return randomPrompt
            } else if (loopMode === "saved prompt") {
                const saved = savedPrompt()
                return saved
            }
        }
        return prompt
    }

    const create = async (looping?: boolean) => {
        const {width, height} = functions.getSizeDimensions(size, getSizeDim())
        const model_name = modelName
        const vae_name = vaeName
        const clip_skip = clipSkip
        const negative_prompt = negativePrompt
        const prompt = await getPrompt(looping)
        const textual_inversions = getActiveTextualInversions()
        const hypernetworks = getActiveHypernetworks()
        const loras = getActiveLoRAs()
        const activeMask = await getActiveMask()
        const activeIPMask = await getActiveIPMask()
        const control_processor = controlProcessor
        const control_scale = controlScale
        const control_start = controlStart
        const control_end = controlEnd
        const guess_mode = controlGuessMode
        const control_reference_image = controlReferenceImage
        const style_fidelity = styleFidelity
        const invisible_watermark = invisibleWatermark
        const novelai_token = novelAIToken
        const holara_cookie = holaraAICookie
        const nsfw_tab = nsfwTab
        const x_adapt_model = xAdaptModel
        const freeu = freeU
        const ip_adapter = ipAdapter
        const ip_processor = ipProcessor
        const ip_weight = ipWeight
        const json = {prompt, negative_prompt, steps, cfg, frames, width, height, denoise, seed, sampler, amount, model_name, vae_name, 
        clip_skip, processing, format, textual_inversions, hypernetworks, loras, control_processor, control_scale, guess_mode,
        control_start, control_end, style_fidelity, control_reference_image, upscaler, nsfw_tab, watermark, invisible_watermark,
        generator, novelai_token, holara_cookie, x_adapt_model, freeu, ip_adapter, ip_processor, ip_weight}
        if (expandImage) json.denoise = 1.0
        const form = new FormData()
        form.append("data", JSON.stringify(json))
        if (imageInput) {
            const image = await getNormalizedImage()
            const arrayBuffer = await fetch(image).then((r) => r.arrayBuffer())
            const blob = new Blob([new Uint8Array(arrayBuffer)])
            const file = new File([blob], "image.png", {type: "image/png"})
            form.append("image", file)
        }
        if (activeMask) {
            const arrayBuffer = await fetch(activeMask).then((r) => r.arrayBuffer())
            const blob = new Blob([new Uint8Array(arrayBuffer)])
            const file = new File([blob], "mask.png", {type: "image/png"})
            form.append("mask", file)
        }
        if (controlImage) {
            let image = controlInvert ? await functions.invertImage(controlImage) : controlImage
            const arrayBuffer = await fetch(image).then((r) => r.arrayBuffer())
            const blob = new Blob([new Uint8Array(arrayBuffer)])
            const file = new File([blob], "control.png", {type: "image/png"})
            form.append("control_image", file)
        }
        if (ipImage) {
            const arrayBuffer = await fetch(ipImage).then((r) => r.arrayBuffer())
            const blob = new Blob([new Uint8Array(arrayBuffer)])
            const file = new File([blob], "image.png", {type: "image/png"})
            form.append("ip_image", file)
        }
        if (activeIPMask) {
            const arrayBuffer = await fetch(activeIPMask).then((r) => r.arrayBuffer())
            const blob = new Blob([new Uint8Array(arrayBuffer)])
            const file = new File([blob], "mask.png", {type: "image/png"})
            form.append("ip_mask", file)
        }
        axios.post("/generate", form)
    }

    const interrupt = async () => {
        axios.post("/interrupt")
    }

    useEffect(() => {
        const json = {infinite}
        axios.post("/update-infinite", json)
    }, [infinite])

    useEffect(() => {
        const json = {upscaling}
        axios.post("/update-upscaling", json)
    }, [upscaling])

    useEffect(() => {
        const json = {precision}
        axios.post("/update-precision", json)
    }, [precision])

    const interrogate = async () => {
        if (!imageInput) return
        const blob = await fetch(imageInput).then((r) => r.blob())
        const file = new File([blob], "image.png", {type: "image/png"})
        const form = new FormData()
        form.append("image", file)
        form.append("model_name", interrogatorName)
        setInterrogateStarted(true)
        const result = await axios.post("/interrogate", form).then((r) => r.data)
        setInterrogateStarted(false)
        if (result) setInterrogateText(result.replaceAll("_", " "))
    }

    const interruptInterrogate = async () => {
        await axios.post("/interrupt-interrogate")
        setInterrogateStarted(false)
    }

    const appendToPrompt = () => {
        if (!interrogateText) return
        const {prompt, steps, cfg, size, clipSkip, denoise, model, vae} = functions.extractMetaValues(interrogateText)
        if (prompt) setPrompt(prompt)
        if (steps) setSteps(Number(steps))
        if (cfg) setCFG(Number(cfg))
        if (clipSkip) setClipSkip(Number(clipSkip))
        if (denoise) setDenoise(Number(denoise))
        if (model) setModelName(model)
        if (vae) setVAEName(vae)
        if (size) setSize(functions.getSizeDimensionsReverse(Number(size.split("x")[1]), getSizeDim()))
        if (functions.isNovelAIModel(model)) {
            if (generator !== "novel ai") setGenerator("novel ai")
        } else if (functions.isHolaraAIModel(model)) {
            if (generator !== "holara ai") setGenerator("holara ai")
        } else {
            if (generator !== "local") setGenerator("local")
        }
    }

    const randomizePrompt = async () => {
        const text = await axios.post("/generate-prompt", {prompt, mode: randomPromptMode}).then((r) => r.data)
        setPrompt(text)
        return text
    }

    const savedPrompt = () => {
        let arr = savedPrompts.filter(Boolean)
        if (generator === "novel ai") arr = savedPromptsNovelAI.filter(Boolean)
        if (generator === "holara ai") arr = savedPromptsHolaraAI.filter(Boolean)
        if (arr.length) {
            let saved = prompt
            while (prompt === saved) {
                saved = arr[Math.floor(Math.random() * arr.length)]
            }
            setPrompt(saved)
            return saved
        }
        return prompt
    }

    useEffect(() => {
        if (!socket) return
        const repeatGeneration = () => {
            create(true)
        }
        socket.on("repeat generation", repeatGeneration)
        return () => {
            socket.off("repeat generation", repeatGeneration)
        }
    }, [socket, loopMode, prompt, negativePrompt, savedPrompts, size, sampler, cfg, modelName, vaeName, savedPromptsNovelAI, 
        savedPromptsHolaraAI, loras, hypernetworks, textualInversions, generator, controlProcessor, styleFidelity])

    const getSizeDim = () => {
        if (generator === "novel ai") return "640"
        if (generator === "holara ai") return "640"
        if (modelName?.includes("XL")) return "1024"
        if (modelName?.includes("SC")) return "1024"
        if (modelName?.includes("V2")) return "768"
        return "512"
    }

    return (
        <div className="generate-bar" onMouseEnter={() => setEnableDrag(false)}>
            <textarea className="generate-bar-textarea" spellCheck={false} value={prompt} onChange={(event) => setPrompt(event.target.value)}></textarea>
            <div className="generate-bar-button-container">
                <div className="generate-bar-create-container">
                    <button className="generate-bar-button" onClick={() => started ? interrupt() : create()} style={{backgroundColor: started ? "var(--buttonBGStop)" : "var(--buttonBG)"}}>{started ? "Stop" : "Create"}</button>
                    <span className="generate-bar-x">x</span>
                    <input className="generate-bar-times" value={amount} onChange={(event) => setAmount(event.target.value)}></input>
                </div>
                <div className="generate-bar-interrogate-container">
                    <div className="generate-bar-loop-container" onClick={() => randomizePrompt()}>
                        <img className="generate-bar-icon" src={diceIcon} style={{filter: getFilter()}}/>
                    </div>
                    <div className="generate-bar-loop-container" onClick={() => savedPrompt()}>
                        <img className="generate-bar-icon" src={bookmarkIcon} style={{filter: getFilter()}}/>
                    </div>
                    <div className="generate-bar-loop-container" onClick={() => appendToPrompt()}>
                        <img className="generate-bar-icon" src={appendToPromptIcon} style={{filter: getFilter()}}/>
                    </div>
                    <div className="generate-bar-loop-container" onClick={() => setInfinite((prev) => !prev)}>
                        <img className="generate-bar-checkbox" src={infinite ? checkboxChecked : checkbox} style={{filter: getFilter()}}/>
                        <img className="generate-bar-icon" src={loop} style={{filter: getFilter()}}/>
                    </div>
                    <div className="generate-bar-loop-container" onClick={() => setUpscaling((prev) => !prev)}>
                        <img className="generate-bar-checkbox" src={upscaling ? checkboxChecked : checkbox} style={{filter: getFilter()}}/>
                        <img className="generate-bar-icon" src={upscale} style={{filter: getFilter()}}/>
                    </div>
                    <button className="generate-bar-button-2" onClick={() => interrogateStarted ? interruptInterrogate() : interrogate()} style={{backgroundColor: interrogateStarted ? "var(--buttonBG)" : "var(--buttonBG2)"}}>{interrogateStarted ? "Stop" : "Interrogate"}</button>
                </div>
            </div>
        </div>
    )
}

export default GenerateBar