import React, {useContext, useEffect, useState, useRef, createElement} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, NegativePromptContext, ProcessingContext,
InterrogatorNameContext, DeletionContext, FormatContext, PrecisionContext, LoopModeContext, WatermarkContext, UpscalerContext, NSFWTabContext,
InvisibleWatermarkContext, SauceNaoAPIKeyContext, RandomPromptModeContext, GeneratorContext, NovelAITokenContext, HolaraAICookieContext, 
ModelDirContext, OutputDirContext, CheckForUpdatesContext} from "../Context"
import functions from "../structures/Functions"
import {Dropdown, DropdownButton} from "react-bootstrap"
import checkbox from "../assets/icons/checkbox2.png"
import checkboxChecked from "../assets/icons/checkbox2-checked.png"
import axios from "axios"
import "./styles/settingsbar.less"

const SettingsBar: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {negativePrompt, setNegativePrompt} = useContext(NegativePromptContext)
    const {interrogatorName, setInterrogatorName} = useContext(InterrogatorNameContext)
    const {upscaler, setUpscaler} = useContext(UpscalerContext)
    const {processing, setProcessing} = useContext(ProcessingContext)
    const {deletion, setDeletion} = useContext(DeletionContext)
    const {precision, setPrecision} = useContext(PrecisionContext)
    const {format, setFormat} = useContext(FormatContext)
    const {loopMode, setLoopMode} = useContext(LoopModeContext)
    const {watermark, setWatermark} = useContext(WatermarkContext)
    const {nsfwTab, setNSFWTab} = useContext(NSFWTabContext)
    const {invisibleWatermark, setInvisibleWatermark} = useContext(InvisibleWatermarkContext)
    const [convertPrompt, setConvertPrompt] = useState("")
    const {saucenaoAPIKey, setSaucenaoAPIKey} = useContext(SauceNaoAPIKeyContext)
    const {randomPromptMode, setRandomPromptMode} = useContext(RandomPromptModeContext)
    const {generator, setGenerator} = useContext(GeneratorContext)
    const {novelAIToken, setNovelAIToken} = useContext(NovelAITokenContext)
    const {holaraAICookie, setHolaraAICookie} = useContext(HolaraAICookieContext)
    const {modelDir, setModelDir} = useContext(ModelDirContext)
    const {outputDir, setOutputDir} = useContext(OutputDirContext)
    const {checkForUpdates, setCheckForUpdates} = useContext(CheckForUpdatesContext)
    const [unloadTrigger, setUnloadTrigger] = useState(false)
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedNegativePrompt = localStorage.getItem("negativePrompt")
        if (savedNegativePrompt?.trim()) setNegativePrompt(savedNegativePrompt)
        const savedInterrogatorName = localStorage.getItem("interrogatorName")
        if (savedInterrogatorName) setInterrogatorName(savedInterrogatorName)
        const savedProcessing = localStorage.getItem("processing")
        if (savedProcessing) setProcessing(savedProcessing)
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
        const savedUpscaler = localStorage.getItem("upscaler")
        if (savedUpscaler) setUpscaler(savedUpscaler)
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
        const savedModelDir = localStorage.getItem("modelDir")
        if (savedModelDir) setModelDir(savedModelDir)
        const savedOutputDir = localStorage.getItem("outputDir")
        if (savedOutputDir) setOutputDir(savedOutputDir)
        const savedCheckForUpdates = localStorage.getItem("checkForUpdates")
        if (savedCheckForUpdates) setCheckForUpdates(savedCheckForUpdates === "true")
    }, [])

    useEffect(() => {
        if (negativePrompt) localStorage.setItem("negativePrompt", String(negativePrompt))
        localStorage.setItem("interrogatorName", String(interrogatorName))
        localStorage.setItem("processing", String(processing))
        localStorage.setItem("format", String(format))
        localStorage.setItem("deletion", String(deletion))
        localStorage.setItem("precision", String(precision))
        localStorage.setItem("loopMode", String(loopMode))
        localStorage.setItem("watermark", String(watermark))
        localStorage.setItem("upscaler", String(upscaler))
        localStorage.setItem("nsfwTab", String(nsfwTab))
        localStorage.setItem("invisibleWatermark", String(invisibleWatermark))
        localStorage.setItem("saucenaoAPIKey", String(saucenaoAPIKey))
        localStorage.setItem("randomPromptMode", String(randomPromptMode))
        localStorage.setItem("generator", String(generator))
        localStorage.setItem("novelAIToken", String(novelAIToken))
        localStorage.setItem("holaraAICookie", String(holaraAICookie))
        localStorage.setItem("modelDir", String(modelDir))
        localStorage.setItem("outputDir", String(outputDir))
        localStorage.setItem("checkForUpdates", String(checkForUpdates))
    }, [negativePrompt, interrogatorName, processing, format, deletion, precision, loopMode, 
        upscaler, nsfwTab, invisibleWatermark, watermark, saucenaoAPIKey, randomPromptMode,
        generator, novelAIToken, holaraAICookie, modelDir, outputDir, checkForUpdates])

    const resetNegativePrompt = () => {
        setNegativePrompt("")
    }

    const getInterrogatorName = () => {
        if (interrogatorName === "wdtagger") return "WDTagger"
        if (interrogatorName === "deepbooru") return "DeepBooru"
        if (interrogatorName === "blip") return "BLIP"
    }

    const getUpscalerName = () => {
        if (upscaler === "waifu2x") return "waifu2x"
        if (upscaler === "real-esrgan") return "Real-ESRGAN"
        if (upscaler === "real-cugan") return "Real-CUGAN"
    }

    const getGenerator = () => {
        if (generator === "local") return "Local"
        if (generator === "novel ai") return "Novel AI"
        if (generator === "holara ai") return "Holara AI"
    }

    const unloadModels = () => {
        setUnloadTrigger(true)
        axios.post("/unload-models")
        setTimeout(() => {
            setUnloadTrigger(false)
        }, 1000)
    }

    return (
        <div className="settings-bar" onMouseEnter={() => setEnableDrag(false)}>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Generator:</span>
                <DropdownButton title={getGenerator()} drop="down" className="checkpoint-selector">
                    <Dropdown.Item active={generator === "local"} onClick={() => setGenerator("local")}>Local</Dropdown.Item>
                    <Dropdown.Item active={generator === "novel ai"} onClick={() => setGenerator("novel ai")}>Novel AI</Dropdown.Item>
                    <Dropdown.Item active={generator === "holara ai"} onClick={() => setGenerator("holara ai")}>Holara AI</Dropdown.Item>
                </DropdownButton>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Processing:</span>
                <DropdownButton title={processing.toUpperCase()} drop="down" className="checkpoint-selector">
                    <Dropdown.Item active={processing === "gpu"} onClick={() => setProcessing("gpu")}>GPU</Dropdown.Item>
                    <Dropdown.Item active={processing === "cpu"} onClick={() => setProcessing("cpu")}>CPU</Dropdown.Item>
                </DropdownButton>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Precision:</span>
                <DropdownButton title={functions.toProperCase(precision)} drop="down" className="checkpoint-selector">
                    <Dropdown.Item active={precision === "full"} onClick={() => setPrecision("full")}>Full</Dropdown.Item>
                    <Dropdown.Item active={precision === "half"} onClick={() => setPrecision("half")}>Half</Dropdown.Item>
                </DropdownButton>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Model Dir:</span>
                <input className="settings-bar-input" spellCheck={false} value={modelDir} onChange={(event) => setModelDir(event.target.value)}/>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Output Dir:</span>
                <input className="settings-bar-input" spellCheck={false} value={outputDir} onChange={(event) => setOutputDir(event.target.value)}/>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Interrogator:</span>
                <DropdownButton title={getInterrogatorName()} drop="down" className="checkpoint-selector">
                    <Dropdown.Item active={interrogatorName === "wdtagger"} onClick={() => setInterrogatorName("wdtagger")}>WDTagger</Dropdown.Item>
                    <Dropdown.Item active={interrogatorName === "deepbooru"} onClick={() => setInterrogatorName("deepbooru")}>DeepBooru</Dropdown.Item>
                    <Dropdown.Item active={interrogatorName === "blip"} onClick={() => setInterrogatorName("blip")}>BLIP</Dropdown.Item>
                </DropdownButton>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Upscaler:</span>
                <DropdownButton title={getUpscalerName()} drop="down" className="checkpoint-selector">
                    <Dropdown.Item active={upscaler === "waifu2x"} onClick={() => setUpscaler("waifu2x")}>waifu2x</Dropdown.Item>
                    <Dropdown.Item active={upscaler === "real-esrgan"} onClick={() => setUpscaler("real-esrgan")}>Real-ESRGAN</Dropdown.Item>
                    <Dropdown.Item active={upscaler === "real-cugan"} onClick={() => setUpscaler("real-cugan")}>Real-CUGAN</Dropdown.Item>
                </DropdownButton>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Loop Mode:</span>
                <DropdownButton title={functions.toProperCase(loopMode)} drop="down" className="checkpoint-selector">
                    <Dropdown.Item active={loopMode === "repeat prompt"} onClick={() => setLoopMode("repeat prompt")}>Repeat Prompt</Dropdown.Item>
                    <Dropdown.Item active={loopMode === "random prompt"} onClick={() => setLoopMode("random prompt")}>Random Prompt</Dropdown.Item>
                    <Dropdown.Item active={loopMode === "saved prompt"} onClick={() => setLoopMode("saved prompt")}>Saved Prompt</Dropdown.Item>
                </DropdownButton>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Format:</span>
                <DropdownButton title={format.toUpperCase()} drop="down" className="checkpoint-selector">
                    <Dropdown.Item active={format === "jpg"} onClick={() => setFormat("jpg")}>JPG</Dropdown.Item>
                    <Dropdown.Item active={format === "png"} onClick={() => setFormat("png")}>PNG</Dropdown.Item>
                    <Dropdown.Item active={format === "webp"} onClick={() => setFormat("webp")}>WEBP</Dropdown.Item>
                    <Dropdown.Item active={format === "gif"} onClick={() => setFormat("gif")}>GIF</Dropdown.Item>
                </DropdownButton>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Watermark:</span>
                <DropdownButton title={watermark ? "Yes" : "No"} drop="down" className="checkpoint-selector">
                    <Dropdown.Item active={watermark === true} onClick={() => setWatermark(true)}>Yes</Dropdown.Item>
                    <Dropdown.Item active={watermark === false} onClick={() => setWatermark(false)}>No</Dropdown.Item>
                </DropdownButton>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Invisible Watermark:</span>
                <DropdownButton title={invisibleWatermark ? "Yes" : "No"} drop="down" className="checkpoint-selector">
                    <Dropdown.Item active={invisibleWatermark === true} onClick={() => setInvisibleWatermark(true)}>Yes</Dropdown.Item>
                    <Dropdown.Item active={invisibleWatermark === false} onClick={() => setInvisibleWatermark(false)}>No</Dropdown.Item>
                </DropdownButton>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Prompt Gen:</span>
                <DropdownButton title={randomPromptMode} drop="down" className="checkpoint-selector">
                    <Dropdown.Item active={randomPromptMode === "1girl"} onClick={() => setRandomPromptMode("1girl")}>1girl</Dropdown.Item>
                    <Dropdown.Item active={randomPromptMode === "1boy"} onClick={() => setRandomPromptMode("1boy")}>1boy</Dropdown.Item>
                    <Dropdown.Item active={randomPromptMode === "custom"} onClick={() => setRandomPromptMode("custom")}>custom</Dropdown.Item>
                </DropdownButton>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">R18:</span>
                <DropdownButton title={nsfwTab ? "Yes" : "No"} drop="down" className="checkpoint-selector">
                    <Dropdown.Item active={nsfwTab === true} onClick={() => setNSFWTab(true)}>Yes</Dropdown.Item>
                    <Dropdown.Item active={nsfwTab === false} onClick={() => setNSFWTab(false)}>No</Dropdown.Item>
                </DropdownButton>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Deletion:</span>
                <DropdownButton title={deletion} drop="down" className="checkpoint-selector">
                    <Dropdown.Item active={deletion === "trash"} onClick={() => setDeletion("trash")}>trash</Dropdown.Item>
                    <Dropdown.Item active={deletion === "permanent"} onClick={() => setDeletion("permanent")}>permanent</Dropdown.Item>
                </DropdownButton>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Check For Updates:</span>
                <DropdownButton title={checkForUpdates ? "Yes" : "No"} drop="down" className="checkpoint-selector">
                    <Dropdown.Item active={checkForUpdates} onClick={() => setCheckForUpdates(true)}>Yes</Dropdown.Item>
                    <Dropdown.Item active={!checkForUpdates} onClick={() => setCheckForUpdates(false)}>No</Dropdown.Item>
                </DropdownButton>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Negative Prompt:</span>
                <textarea className="settings-bar-textarea" spellCheck={false} value={negativePrompt} onChange={(event) => setNegativePrompt(event.target.value)}></textarea>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Convert Prompt:</span>
                <textarea className="settings-bar-textarea" spellCheck={false} value={convertPrompt} onChange={(event) => setConvertPrompt(event.target.value)}></textarea>
                <button className="settings-bar-button" onClick={() => setConvertPrompt(functions.convertPrompt(convertPrompt))}>Convert</button>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">SauceNao Key:</span>
                <input className="settings-bar-input" spellCheck={false} value={saucenaoAPIKey} onChange={(event) => setSaucenaoAPIKey(event.target.value)}/>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Novel AI Token:</span>
                <input className="settings-bar-input" spellCheck={false} value={novelAIToken} onChange={(event) => setNovelAIToken(event.target.value)}/>
            </div>
            <div className="settings-bar-row">
                <span className="settings-bar-text">Holara AI Cookie:</span>
                <input className="settings-bar-input" spellCheck={false} value={holaraAICookie} onChange={(event) => setHolaraAICookie(event.target.value)}/>
            </div>
            <div className="settings-bar-row">
                <button className="settings-bar-button" onClick={() => unloadModels()} style={{backgroundColor: unloadTrigger ? "var(--buttonBGStop)" : "var(--buttonBG)"}}>{unloadTrigger ? "Done!" : "Unload Models"}</button>
            </div>
        </div>
    )
}

export default SettingsBar