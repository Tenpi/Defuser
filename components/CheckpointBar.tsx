import React, {useContext, useEffect, useState, useRef} from "react"
import {useHistory} from "react-router-dom"
import {HashLink as Link} from "react-router-hash-link"
import favicon from "../assets/icons/favicon.png"
import {EnableDragContext, MobileContext, SiteHueContext, SiteSaturationContext, SiteLightnessContext, ModelNameContext, ModelNamesContext,
VAENameContext, VAENamesContext, TabContext, GeneratorContext, ThemeContext, ThemeSelectorContext} from "../Context"
import functions from "../structures/Functions"
import {Dropdown, DropdownButton} from "react-bootstrap"
import checkpoint from "../assets/icons/checkpoint.png"
import vae from "../assets/icons/vae.png"
import generate from "../assets/icons/generate.png"
import train from "../assets/icons/train.png"
import misc from "../assets/icons/misc.png"
import settings from "../assets/icons/settings.png"
import view from "../assets/icons/view.png"
import savedPrompts from "../assets/icons/saved-prompts.png"
import watermark from "../assets/icons/watermark.png"
import classify from "../assets/icons/classify.png"
import "./styles/checkpointbar.less"
import axios from "axios"

const CheckpointBar: React.FunctionComponent = (props) => {
    const {theme, getTheme} = useContext(ThemeContext)
    const {themeSelector, getThemeSelector} = useContext(ThemeSelectorContext)
    const {enableDrag, setEnableDrag} = useContext(EnableDragContext)
    const {mobile, setMobile} = useContext(MobileContext)
    const {siteHue, setSiteHue} = useContext(SiteHueContext)
    const {siteSaturation, setSiteSaturation} = useContext(SiteSaturationContext)
    const {siteLightness, setSiteLightness} = useContext(SiteLightnessContext)
    const {modelName, setModelName} = useContext(ModelNameContext)
    const {modelNames, setModelNames} = useContext(ModelNamesContext)
    const {vaeName, setVAEName} = useContext(VAENameContext)
    const {vaeNames, setVAENames} = useContext(VAENamesContext)
    const {tab, setTab} = useContext(TabContext)
    const {generator, setGenerator} = useContext(GeneratorContext)
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
        const savedModelName = localStorage.getItem("modelName")
        if (savedModelName) setModelName(savedModelName)
        const savedVAEName = localStorage.getItem("vaeName")
        if (savedVAEName) setVAEName(savedVAEName)
    }, [])

    useEffect(() => {
        localStorage.setItem("modelName", String(modelName))
        localStorage.setItem("vaeName", String(vaeName))
    }, [modelName, vaeName])

    const updateModelNames = async (first?: boolean) => {
        if (generator === "novel ai") {
            const modelNames = ["nai-diffusion-3", "nai-diffusion-2", "nai-diffusion", "safe-diffusion"]
            setModelNames(modelNames)
            if (first) setModelName(modelNames[0])
            if (!modelNames.includes(modelName)) setModelName(modelNames[0])
        } else if (generator === "holara ai") {
            const modelNames = ["Aika", "Akasha", "Vibrance", "Aurora", "Chroma", "Tranquility", "Yami"]
            setModelNames(modelNames)
            if (first) setModelName(modelNames[0])
            if (!modelNames.includes(modelName)) setModelName(modelNames[0])
        } else {
            const modelNames = await axios.get("/diffusion-models").then((r) => r.data)
            setModelNames(modelNames)
            if (first) setModelName(modelNames[0])
            if (!modelNames.includes(modelName)) setModelName(modelNames[0])
        }
    }

    useEffect(() => {
        if (!modelName && !vaeName) return
        if (generator) {
            setTimeout(() => {
                updateModelNames()
                updateVAENames()
            }, 200)
        }
    }, [generator, modelName, vaeName])

    const modelsJSX = () => {
        let jsx = [] as any
        for (let i = 0; i < modelNames.length; i++) {
            jsx.push(<Dropdown.Item active={modelName === modelNames[i]} onClick={() => setModelName(modelNames[i])}>{modelNames[i]}</Dropdown.Item>)
        }
        return jsx 
    }

    const updateVAENames = async (first?: boolean) => {
        if (generator === "novel ai" || generator === "holara ai") {
            const vaeNames = ["None"]
            setVAENames(vaeNames)
            if (first) setVAEName(vaeNames[0])
            if (!vaeNames.includes(vaeName)) setVAEName(vaeNames[0])
        } else {
            const vaeNames = await axios.get("/vae-models").then((r) => r.data)
            setVAENames(vaeNames)
            if (first) setVAEName(vaeNames[0])
            if (!vaeNames.includes(vaeName)) setVAEName(vaeNames[0])
        }
    }

    const vaesJSX = () => {
        let jsx = [] as any
        for (let i = 0; i < vaeNames.length; i++) {
            jsx.push(<Dropdown.Item active={vaeName === vaeNames[i]} onClick={() => setVAEName(vaeNames[i])}>{vaeNames[i]}</Dropdown.Item>)
        }
        return jsx 
    }

    return (
        <div className="checkpoint-bar" onMouseEnter={() => setEnableDrag(false)}>
            <img className="checkpoint-bar-icon" src={checkpoint} style={{cursor: "default", filter: getFilter()}}/>
            <DropdownButton title={modelName} drop="down" className="checkpoint-selector" onClick={() => updateModelNames()}>
                {modelsJSX()}
            </DropdownButton>
            <img className="checkpoint-bar-icon" src={vae} style={{marginLeft: "10px", cursor: "default", filter: getFilter()}}/>
            <DropdownButton title={vaeName} drop="down" className="checkpoint-selector" onClick={() => updateVAENames()}>
                {vaesJSX()}
            </DropdownButton>
            <img className="checkpoint-bar-icon" src={generate} style={{marginLeft: "10px", cursor: "pointer", filter: getFilter()}} onClick={() => setTab("generate")}/>
            <img className="checkpoint-bar-icon" src={view} style={{filter: getFilter(), cursor: "pointer"}} onClick={() => setTab("view")}/>
            <img className="checkpoint-bar-icon" src={savedPrompts} style={{filter: getFilter(), cursor: "pointer"}} onClick={() => setTab("saved prompts")}/>
            <img className="checkpoint-bar-icon" src={watermark} style={{filter: getFilter(), cursor: "pointer"}} onClick={() => setTab("watermark")}/>
            <img className="checkpoint-bar-icon" src={train} style={{filter: getFilter(), cursor: "pointer"}} onClick={() => setTab("train")}/>
            <img className="checkpoint-bar-icon" src={classify} style={{filter: getFilter(), cursor: "pointer"}} onClick={() => setTab("classify")}/>
            <img className="checkpoint-bar-icon" src={misc} style={{filter: getFilter(), cursor: "pointer"}} onClick={() => setTab("misc")}/>
            <img className="checkpoint-bar-icon" src={settings} style={{filter: getFilter(), cursor: "pointer"}} onClick={() => setTab("settings")}/>
        </div>
    )
}

export default CheckpointBar