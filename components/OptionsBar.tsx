import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, StepsContext,
CFGContext, SizeContext, DenoiseContext, SamplerContext, SeedContext, InterrogateTextContext, ClipSkipContext} from "../Context"
import functions from "../structures/Functions"
import {Dropdown, DropdownButton} from "react-bootstrap"
import stepsIcon from "../assets/icons/steps.png"
import cfgIcon from "../assets/icons/cfg.png"
import sizeIcon from "../assets/icons/size.png"
import denoiseIcon from "../assets/icons/denoise.png"
import seedIcon from "../assets/icons/seed.png"
import clipSkipIcon from "../assets/icons/clipskip.png"
import Slider from "react-slider"
import OptionsImage from "./OptionsImage"
import "./styles/optionsbar.less"

const OptionsBar: React.FunctionComponent = (props) => {
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {steps, setSteps} = useContext(StepsContext)
    const {cfg, setCFG} = useContext(CFGContext)
    const {size, setSize} = useContext(SizeContext)
    const {denoise, setDenoise} = useContext(DenoiseContext)
    const {seed, setSeed} = useContext(SeedContext)
    const {sampler, setSampler} = useContext(SamplerContext)
    const {clipSkip, setClipSkip} = useContext(ClipSkipContext)
    const {interrogateText, setInterrogateText} = useContext(InterrogateTextContext)
    const ref = useRef<HTMLCanvasElement>(null)
    const history = useHistory()

    const getFilter = () => {
        return `hue-rotate(${siteHue - 180}deg) saturate(${siteSaturation}%) brightness(${siteLightness + 50}%)`
    }

    useEffect(() => {
        const savedSteps = localStorage.getItem("steps")
        if (savedSteps) setSteps(Number(savedSteps))
        const savedCFG = localStorage.getItem("cfg")
        if (savedCFG) setCFG(Number(savedCFG))
        const savedSize = localStorage.getItem("size")
        if (savedSize) setSize(Number(savedSize))
        const savedDenoise = localStorage.getItem("denoise")
        if (savedDenoise) setDenoise(Number(savedDenoise))
        const savedSeed = localStorage.getItem("seed")
        if (savedSeed) setSeed(Number(savedSeed))
        const savedSampler = localStorage.getItem("sampler")
        if (savedSampler) setSampler(savedSampler)
        const savedClipSkip = localStorage.getItem("clipSkip")
        if (savedClipSkip) setClipSkip(savedClipSkip)
        const savedInterrogateText = localStorage.getItem("interrogateText")
        if (savedInterrogateText) setInterrogateText(savedInterrogateText)
    }, [])

    useEffect(() => {
        localStorage.setItem("steps", String(steps))
        localStorage.setItem("cfg", String(cfg))
        localStorage.setItem("size", String(size))
        localStorage.setItem("denoise", String(denoise))
        localStorage.setItem("seed", String(seed))
        localStorage.setItem("sampler", String(sampler))
        localStorage.setItem("clipSkip", String(clipSkip))
        localStorage.setItem("interrogateText", String(interrogateText))
    }, [steps, cfg, size, denoise, seed, sampler, clipSkip, interrogateText])

    const getSampler = () => {
        if (sampler === "euler a") return "Euler A"
        if (sampler === "euler") return "Euler"
        if (sampler === "unipc") return "UniPC"
        return sampler.toUpperCase()
    }

    return (
        <div className="options-bar" onMouseEnter={() => setEnableDrag(false)}>
            <div className="options-bar-img-input">
                    <span className="options-bar-text">Img Input</span>
                    <OptionsImage/>
            </div>
            <div className="options-bar-options">
                <div className="options-bar-option-row">
                    <span className="options-option-text">Steps</span>
                    <img className="options-option-icon" src={stepsIcon} style={{filter: getFilter()}}/>
                    <Slider className="options-slider" trackClassName="options-slider-track" thumbClassName="options-slider-thumb" onChange={(value) => setSteps(value)} min={1} max={100} step={1} value={steps}/>
                    <span className="options-option-text-value">{steps}</span>
                </div>
                <div className="options-bar-option-row">
                    <span className="options-option-text">CFG</span>
                    <img className="options-option-icon" src={cfgIcon} style={{filter: getFilter()}}/>
                    <Slider className="options-slider" trackClassName="options-slider-track" thumbClassName="options-slider-thumb" onChange={(value) => setCFG(value)} min={0} max={30} step={1} value={cfg}/>
                    <span className="options-option-text-value">{cfg}</span>
                </div>
                <div className="options-bar-option-row">
                    <span className="options-option-text">Size</span>
                    <img className="options-option-icon" src={sizeIcon} style={{filter: getFilter()}}/>
                    <Slider className="options-slider-small" trackClassName="options-slider-track" thumbClassName="options-slider-thumb" onChange={(value) => setSize(value)} min={-1} max={1} step={0.125} value={size}/>
                    <span className="options-option-text-value">{functions.getSizeDimensions(size).width}x{functions.getSizeDimensions(size).height}</span>
                </div>
                <div className="options-bar-option-row">
                    <span className="options-option-text">Denoise</span>
                    <img className="options-option-icon" src={denoiseIcon} style={{filter: getFilter()}}/>
                    <Slider className="options-slider" trackClassName="options-slider-track" thumbClassName="options-slider-thumb" onChange={(value) => setDenoise(value)} min={0.01} max={1} step={0.01} value={denoise}/>
                    <span className="options-option-text-value">{denoise}</span>
                </div>
                <div className="options-bar-option-row">
                    <span className="options-option-text">Clip Skip</span>
                    <img className="options-option-icon" src={clipSkipIcon} style={{filter: getFilter()}}/>
                    <Slider className="options-slider" trackClassName="options-slider-track" thumbClassName="options-slider-thumb" onChange={(value) => setClipSkip(value)} min={1} max={4} step={1} value={clipSkip}/>
                    <span className="options-option-text-value">{clipSkip}</span>
                </div>
                <div className="options-bar-option-row">
                    <span className="options-option-text">Seed</span>
                    <img className="options-option-icon" src={seedIcon} style={{filter: getFilter()}}/>
                    <input className="options-option-input" spellCheck={false} value={seed} onChange={(event) => setSeed(event.target.value)} style={{width: "150px"}}></input>
                    <span className="options-option-text" style={{marginRight: "10px"}}>Sampler</span>
                    <DropdownButton title={getSampler()} drop="down" className="checkpoint-selector">
                        <Dropdown.Item active={sampler === "euler a"} onClick={() => setSampler("euler a")}>Euler A</Dropdown.Item>
                        <Dropdown.Item active={sampler === "euler"} onClick={() => setSampler("euler")}>Euler</Dropdown.Item>
                        <Dropdown.Item active={sampler === "dpm++"} onClick={() => setSampler("dpm++")}>DPM++</Dropdown.Item>
                        <Dropdown.Item active={sampler === "ddim"} onClick={() => setSampler("ddim")}>DDIM</Dropdown.Item>
                        <Dropdown.Item active={sampler === "ddpm"} onClick={() => setSampler("ddpm")}>DDPM</Dropdown.Item>
                        <Dropdown.Item active={sampler === "unipc"} onClick={() => setSampler("unipc")}>UniPC</Dropdown.Item>
                        <Dropdown.Item active={sampler === "deis"} onClick={() => setSampler("deis")}>DEIS</Dropdown.Item>
                        <Dropdown.Item active={sampler === "heun"} onClick={() => setSampler("heun")}>HEUN</Dropdown.Item>
                    </DropdownButton>
                </div>
            </div>
            <div className="options-bar-interrogator">
                    <textarea className="options-option-textarea" spellCheck={false} disabled={true} value={interrogateText} onChange={(event) => setInterrogateText(event.target.value)}></textarea>
            </div>
        </div>
    )
}

export default OptionsBar