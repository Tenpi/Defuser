import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, ControlProcessorContext, 
ThemeContext, ImageInputContext, ControlImageContext, ControlScaleContext, ControlGuessModeContext, ControlStartContext, ControlEndContext, 
ControlInvertContext, StyleFidelityContext, ControlReferenceImageContext, ImageBrightnessContext, ImageContrastContext,
ExpandImageContext, UpscalerContext, ImageHueContext, ImageSaturationContext} from "../Context"
import functions from "../structures/Functions"
import Slider from "react-slider"
import radioButtonOff from "../assets/icons/radiobutton-off.png"
import radioButtonOn from "../assets/icons/radiobutton-on.png"
import radioButtonOffLight from "../assets/icons/radiobutton-off-light.png"
import radioButtonOnLight from "../assets/icons/radiobutton-on-light.png"
import checkboxChecked from "../assets/icons/checkbox-checked.png"
import checkbox from "../assets/icons/checkbox.png"
import alphaIcon from "../assets/icons/segmentate.png"
import downloadIcon from "../assets/icons/download.png"
import "./styles/controlnet.less"
import axios from "axios"

const ControlNet: React.FunctionComponent = (props) => {
    const {theme, setTheme} = useContext(ThemeContext)
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {imageBrightness, setImageBrightness} = useContext(ImageBrightnessContext)
    const {imageContrast, setImageContrast} = useContext(ImageContrastContext)
    const {imageHue, setImageHue} = useContext(ImageHueContext)
    const {imageSaturation, setImageSaturation} = useContext(ImageSaturationContext)
    const {controlProcessor, setControlProcessor} = useContext(ControlProcessorContext)
    const {imageInput, setImageInput} = useContext(ImageInputContext)
    const {controlImage, setControlImage} = useContext(ControlImageContext)
    const {controlScale, setControlScale} = useContext(ControlScaleContext)
    const {controlGuessMode, setControlGuessMode} = useContext(ControlGuessModeContext)
    const {controlStart, setControlStart} = useContext(ControlStartContext)
    const {controlEnd, setControlEnd} = useContext(ControlEndContext)
    const {controlInvert, setControlInvert} = useContext(ControlInvertContext)
    const {controlReferenceImage, setControlReferenceImage} = useContext(ControlReferenceImageContext)
    const {expandImage, setExpandImage} = useContext(ExpandImageContext)
    const [imageHover, setImageHover] = useState(false)
    const {upscaler, setUpscaler} = useContext(UpscalerContext)
    const {styleFidelity, setStyleFidity} = useContext(StyleFidelityContext)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    const getFilter2 = () => {
        if (typeof window === "undefined") return
        const bodyStyles = window.getComputedStyle(document.body)
        const color = bodyStyles.getPropertyValue("--text")
        return functions.calculateFilter(color)
    }

    useEffect(() => {
        const savedControlProcessor = localStorage.getItem("controlProcessor")
        if (savedControlProcessor) setControlProcessor(savedControlProcessor)
        const savedControlScale = localStorage.getItem("controlScale")
        if (savedControlScale) setControlScale(Number(savedControlScale))
        const savedControlGuessMode = localStorage.getItem("controlGuessMode")
        if (savedControlGuessMode) setControlGuessMode(savedControlGuessMode === "true")
        const savedControlStart = localStorage.getItem("controlStart")
        if (savedControlStart) setControlStart(Number(savedControlStart))
        const savedControlEnd = localStorage.getItem("controlEnd")
        if (savedControlEnd) setControlEnd(Number(savedControlEnd))
        const savedControlInvert = localStorage.getItem("controlInvert")
        if (savedControlInvert) setControlInvert(savedControlInvert === "true")
        const savedStyleFidelity = localStorage.getItem("styleFidelity")
        if (savedStyleFidelity) setStyleFidity(Number(savedStyleFidelity))
        const savedControlReferenceImage = localStorage.getItem("controlReferenceImage")
        if (savedControlReferenceImage) setControlReferenceImage(savedControlReferenceImage === "true")
    }, [])

    useEffect(() => {
        localStorage.setItem("controlProcessor", String(controlProcessor))
        localStorage.setItem("controlScale", String(controlScale))
        localStorage.setItem("controlGuessMode", String(controlGuessMode))
        localStorage.setItem("controlStart", String(controlStart))
        localStorage.setItem("controlEnd", String(controlEnd))
        localStorage.setItem("controlInvert", String(controlInvert))
        localStorage.setItem("styleFidelity", String(styleFidelity))
        localStorage.setItem("controlReferenceImage", String(controlReferenceImage))
    }, [controlProcessor, controlScale, controlGuessMode, controlStart, controlEnd, controlInvert, styleFidelity, controlReferenceImage])

    const getRadioButton = (condition: boolean) => {
        if (theme === "light") {
            return condition ? radioButtonOnLight : radioButtonOffLight
        } else {
            return condition ? radioButtonOn : radioButtonOff
        }
    }

    const getControlImage = async (upscale?: boolean, alpha?: boolean) => {
        const form = new FormData()
        const arrayBuffer = await fetch(expandImage ? expandImage : imageInput).then((r) => r.arrayBuffer())
        const blob = new Blob([new Uint8Array(arrayBuffer)])
        const file = new File([blob], "image.png", {type: "image/png"})
        form.append("image", file)
        form.append("processor", controlProcessor)
        if (upscale) {
            form.append("upscale", "true")
            form.append("upscaler", upscaler)
            form.append("invert", String(controlInvert))
            form.append("alpha", alpha ? "true" : "false")
        }
        const controlBlob = await axios.post("/control-image", form, {responseType: "blob"}).then((r) => r.data)
        const controlImage = URL.createObjectURL(controlBlob)
        return controlImage
    }

    const updateControlNetImage = async () => {
        if (!imageInput || controlProcessor === "none") return setControlImage("")
        const controlImage = await getControlImage()
        setControlImage(controlImage)
    }

    useEffect(() => {
        updateControlNetImage()
    }, [controlProcessor, imageInput, expandImage])

    const downloadAlpha = async () => {
        if (!imageInput || controlProcessor === "none") return
        await getControlImage(true, true)
    }

    const download = async () => {
        if (!imageInput || controlProcessor === "none") return
        await getControlImage(true, false)
    }

    const isLineArt = () => {
        if (controlProcessor === "depth") return false
        if (controlProcessor === "reference") return false
        return true
    }

    const controlNetOptionsJSX = () => {
        if (controlProcessor === "reference") {
            return (
                <div className="control-image-options-container">
                    <div className="control-option-row">
                        <span className="control-option-text">Invert?</span>
                        <img className="control-checkbox" src={controlInvert ? checkboxChecked : checkbox} onClick={() => setControlInvert((prev: boolean) => !prev)} style={{filter: getFilter2()}}/>
                    </div>
                    <div className="control-option-row">
                        <span className="control-option-text">Guess Mode?</span>
                        <img className="control-checkbox" src={controlGuessMode ? checkboxChecked : checkbox} onClick={() => setControlGuessMode((prev: boolean) => !prev)} style={{filter: getFilter2()}}/>
                    </div>
                    <div className="control-option-row">
                        <span className="control-option-text">Control Scale</span>
                        <Slider className="control-slider" trackClassName="control-slider-track" thumbClassName="control-slider-thumb" onChange={(value) => setControlScale(value)} min={0} max={1} step={0.01} value={controlScale}/>
                        <span className="control-option-text-value">{controlScale}</span>
                    </div>
                    <div className="control-option-row">
                        <span className="control-option-text">Style Fidelity</span>
                        <Slider className="control-slider" trackClassName="control-slider-track" thumbClassName="control-slider-thumb" onChange={(value) => setStyleFidity(value)} min={0} max={1} step={0.01} value={styleFidelity}/>
                        <span className="control-option-text-value">{styleFidelity}</span>
                    </div>
                </div>
            )
        }
        return (
            <div className="control-image-options-container">
                <div className="control-option-row">
                    <span className="control-option-text">Reference Image?</span>
                    <img className="control-checkbox" src={controlReferenceImage ? checkboxChecked : checkbox} onClick={() => setControlReferenceImage((prev: boolean) => !prev)} style={{filter: getFilter2()}}/>
                </div>
                <div className="control-option-row">
                    <span className="control-option-text">Invert?</span>
                    <img className="control-checkbox" src={controlInvert ? checkboxChecked : checkbox} onClick={() => setControlInvert((prev: boolean) => !prev)} style={{filter: getFilter2()}}/>
                </div>
                <div className="control-option-row">
                    <span className="control-option-text">Guess Mode?</span>
                    <img className="control-checkbox" src={controlGuessMode ? checkboxChecked : checkbox} onClick={() => setControlGuessMode((prev: boolean) => !prev)} style={{filter: getFilter2()}}/>
                </div>
                <div className="control-option-row">
                    <span className="control-option-text">Control Scale</span>
                    <Slider className="control-slider" trackClassName="control-slider-track" thumbClassName="control-slider-thumb" onChange={(value) => setControlScale(value)} min={0} max={1} step={0.01} value={controlScale}/>
                    <span className="control-option-text-value">{controlScale}</span>
                </div>
                <div className="control-option-row">
                    <span className="control-option-text">Control Start</span>
                    <Slider className="control-slider" trackClassName="control-slider-track" thumbClassName="control-slider-thumb" onChange={(value) => setControlStart(value)} min={0} max={1} step={0.01} value={controlStart}/>
                    <span className="control-option-text-value">{controlStart}</span>
                </div>
                <div className="control-option-row">
                    <span className="control-option-text">Control End</span>
                    <Slider className="control-slider" trackClassName="control-slider-track" thumbClassName="control-slider-thumb" onChange={(value) => setControlEnd(value)} min={0} max={1} step={0.01} value={controlEnd}/>
                    <span className="control-option-text-value">{controlEnd}</span>
                </div>
            </div>
        )
    }

    return (
        <div className="controlnet">
            <div className="controlnet-container">
                <div className="controlnet-title">ControlNet</div>
                <div className="controlnet-buttons-container">
                    <div className="controlnet-button-container" onClick={() => setControlProcessor("none")}>
                        <img className="controlnet-radio-button" src={getRadioButton(controlProcessor === "none")} style={{filter: getFilter()}}/>
                        <button className="controlnet-button">None</button>
                    </div>
                    <div className="controlnet-button-container" onClick={() => setControlProcessor("canny")}>
                        <img className="controlnet-radio-button" src={getRadioButton(controlProcessor === "canny")} style={{filter: getFilter()}}/>
                        <button className="controlnet-button">Canny</button>
                    </div>
                    <div className="controlnet-button-container" onClick={() => setControlProcessor("depth")}>
                        <img className="controlnet-radio-button" src={getRadioButton(controlProcessor === "depth")} style={{filter: getFilter()}}/>
                        <button className="controlnet-button">Depth</button>
                    </div>
                    <div className="controlnet-button-container" onClick={() => setControlProcessor("lineart")}>
                        <img className="controlnet-radio-button" src={getRadioButton(controlProcessor === "lineart")} style={{filter: getFilter()}}/>
                        <button className="controlnet-button">Lineart</button>
                    </div>
                    <div className="controlnet-button-container" onClick={() => setControlProcessor("lineart anime")}>
                        <img className="controlnet-radio-button" src={getRadioButton(controlProcessor === "lineart anime")} style={{filter: getFilter()}}/>
                        <button className="controlnet-button">Anime</button>
                    </div>
                    <div className="controlnet-button-container" onClick={() => setControlProcessor("lineart manga")}>
                        <img className="controlnet-radio-button" src={getRadioButton(controlProcessor === "lineart manga")} style={{filter: getFilter()}}/>
                        <button className="controlnet-button">Manga</button>
                    </div>
                    <div className="controlnet-button-container" onClick={() => setControlProcessor("softedge")}>
                        <img className="controlnet-radio-button" src={getRadioButton(controlProcessor === "softedge")} style={{filter: getFilter()}}/>
                        <button className="controlnet-button">Softedge</button>
                    </div>
                    <div className="controlnet-button-container" onClick={() => setControlProcessor("scribble")}>
                        <img className="controlnet-radio-button" src={getRadioButton(controlProcessor === "scribble")} style={{filter: getFilter()}}/>
                        <button className="controlnet-button">Scribble</button>
                    </div>
                    <div className="controlnet-button-container" onClick={() => setControlProcessor("reference")}>
                        <img className="controlnet-radio-button" src={getRadioButton(controlProcessor === "reference")} style={{filter: getFilter()}}/>
                        <button className="controlnet-button">Reference</button>
                    </div>
                </div>
                {controlImage ? <div className="control-image-drawer">
                    <div className="control-image-drawer-container">
                        <div className="control-image-container" style={{filter: controlInvert ? "invert(1)" : ""}} onMouseEnter={() => setImageHover(true)} onMouseLeave={() => setImageHover(false)}>
                            <div className={`control-image-button-container ${imageHover ? "show-control-image-buttons" : ""}`}>
                                {isLineArt() ? <img className="control-image-button" src={alphaIcon} onClick={downloadAlpha} draggable={false} style={{filter: controlInvert ? "invert(1)" : ""}}/> : null}
                                <img className="control-image-button" src={downloadIcon} onClick={download} draggable={false} style={{filter: controlInvert ? "invert(1)" : ""}}/>
                            </div>
                            <img className="control-image" src={controlImage} draggable={false} style={{filter: `brightness(${imageBrightness + 100}%) contrast(${imageContrast + 100}%) hue-rotate(${imageHue - 180}deg) saturate(${imageSaturation}%)`}}/>
                        </div>
                        {controlNetOptionsJSX()}
                    </div>
                </div> : null}
            </div>
        </div>
    )
}

export default ControlNet