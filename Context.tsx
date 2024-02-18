import React, {useState} from "react"

export const EnableDragContext = React.createContext<any>(null)
export const MobileContext = React.createContext<any>(null)
export const ThemeContext = React.createContext<any>(null)
export const SiteHueContext = React.createContext<any>(null)
export const SiteSaturationContext = React.createContext<any>(null)
export const SiteLightnessContext = React.createContext<any>(null)
export const ImageBrightnessContext = React.createContext<any>(null)
export const ImageContrastContext = React.createContext<any>(null)

export const StepsContext = React.createContext<any>(null)
export const CFGContext = React.createContext<any>(null)
export const SizeContext = React.createContext<any>(null)
export const DenoiseContext = React.createContext<any>(null)
export const SeedContext = React.createContext<any>(null)
export const SamplerContext = React.createContext<any>(null)
export const InterrogateTextContext = React.createContext<any>(null)
export const InterrogatorNameContext = React.createContext<any>(null)
export const PromptContext = React.createContext<any>(null)
export const NegativePromptContext = React.createContext<any>(null)
export const ClipSkipContext = React.createContext<any>(null)
export const AmountContext = React.createContext<any>(null)
export const ModelNameContext = React.createContext<any>(null)
export const ModelNamesContext = React.createContext<any>(null)
export const VAENameContext = React.createContext<any>(null)
export const VAENamesContext = React.createContext<any>(null)
export const RenderImageContext = React.createContext<any>(null)
export const SocketContext = React.createContext<any>(null)
export const ImagesContext = React.createContext<any>(null)
export const NSFWImagesContext = React.createContext<any>(null)
export const ImageInputImagesContext = React.createContext<any>(null)
export const UpdateImagesContext = React.createContext<any>(null)
export const UpdateSavedContext = React.createContext<any>(null)
export const PreviewImageContext = React.createContext<any>(null)
export const TabContext = React.createContext<any>(null)
export const ProcessingContext = React.createContext<any>(null)
export const ImageInputContext = React.createContext<any>(null)
export const DeletionContext = React.createContext<any>(null)
export const FormatContext = React.createContext<any>(null)
export const TextualInversionsContext = React.createContext<any>(null)
export const HypernetworksContext = React.createContext<any>(null)
export const LorasContext = React.createContext<any>(null)
export const DrawImageContext = React.createContext<any>(null)
export const MaskImageContext = React.createContext<any>(null)
export const MaskDataContext = React.createContext<any>(null)
export const ReverseSortContext = React.createContext<any>(null)
export const PrecisionContext = React.createContext<any>(null)
export const SidebarTypeContext = React.createContext<any>(null)
export const ControlProcessorContext = React.createContext<any>(null)
export const ControlImageContext = React.createContext<any>(null)
export const ControlScaleContext = React.createContext<any>(null)
export const ControlGuessModeContext = React.createContext<any>(null)
export const ControlStartContext = React.createContext<any>(null)
export const ControlEndContext = React.createContext<any>(null)
export const ControlInvertContext = React.createContext<any>(null)
export const StyleFidelityContext = React.createContext<any>(null)
export const ControlReferenceImageContext = React.createContext<any>(null)
export const ExpandDialogFlagContext = React.createContext<any>(null)
export const HorizontalExpandContext = React.createContext<any>(null)
export const VerticalExpandContext = React.createContext<any>(null)
export const ExpandImageContext = React.createContext<any>(null)
export const ExpandMaskContext = React.createContext<any>(null)
export const StartedContext = React.createContext<any>(null)
export const LoopModeContext = React.createContext<any>(null)
export const SavedPromptsContext = React.createContext<any>(null)
export const ViewImagesContext = React.createContext<any>(null)
export const ImageNameContext = React.createContext<any>(null)
export const UpscalerContext = React.createContext<any>(null)
export const NSFWTabContext = React.createContext<any>(null)

export const WatermarkContext = React.createContext<any>(null)
export const AIWatermarkPositionContext = React.createContext<any>(null)
export const AIWatermarkTypeContext = React.createContext<any>(null)
export const AIWatermarkHueContext = React.createContext<any>(null)
export const AIWatermarkSaturationContext = React.createContext<any>(null)
export const AIWatermarkBrightnessContext = React.createContext<any>(null)
export const AIWatermarkInvertContext = React.createContext<any>(null)
export const AIWatermarkOpacityContext = React.createContext<any>(null)
export const AIWatermarkMarginXContext = React.createContext<any>(null)
export const AIWatermarkMarginYContext = React.createContext<any>(null)
export const AIWatermarkScaleContext = React.createContext<any>(null)

const Context: React.FunctionComponent = (props: any) => {
    const [theme, setTheme] = useState("light")
    const [siteHue, setSiteHue] = useState(180)
    const [siteSaturation, setSiteSaturation] = useState(100)
    const [siteLightness, setSiteLightness] = useState(50)
    const [imageBrightness, setImageBrightness] = useState(0)
    const [imageContrast, setImageContrast] = useState(0)
    const [steps, setSteps] = useState(20)
    const [cfg, setCFG] = useState(7)
    const [size, setSize] = useState(0)
    const [denoise, setDenoise] = useState(0.5)
    const [seed, setSeed] = useState("-1")
    const [sampler, setSampler] = useState("euler a")
    const [prompt, setPrompt] = useState("")
    const [negativePrompt, setNegativePrompt] = useState("")
    const [clipSkip, setClipSkip] = useState(2)
    const [interrogateText, setInterrogateText] = useState("")
    const [interrogatorName, setInterrogatorName] = useState("wdtagger")
    const [amount, setAmount] = useState("1")
    const [modelName, setModelName] = useState("")
    const [modelNames, setModelNames] = useState([])
    const [vaeName, setVAEName] = useState("")
    const [vaeNames, setVAENames] = useState([])
    const [renderImage, setRenderImage] = useState("")
    const [updateSaved, setUpdateSaved] = useState(false)
    const [previewImage, setPreviewImage] = useState("")
    const [tab, setTab] = useState("generate")
    const [processing, setProcessing] = useState("gpu")
    const [imageInput, setImageInput] = useState("")
    const [deletion, setDeletion] = useState("trash")
    const [format, setFormat] = useState("png")
    const [drawImage, setDrawImage] = useState("")
    const [maskImage, setMaskImage] = useState("")
    const [maskData, setMaskData] = useState("")
    const [precision, setPrecision] = useState("full")
    const [sidebarType, setSidebarType] = useState("history")
    const [controlProcessor, setControlProcessor] = useState("none")
    const [controlImage, setControlImage] = useState("")
    const [controlScale, setControlScale] = useState(1)
    const [controlGuessMode, setControlGuessMode] = useState(false)
    const [controlStart, setControlStart] = useState(0)
    const [controlEnd, setControlEnd] = useState(1)
    const [controlInvert, setControlInvert] = useState(false)
    const [styleFidelity, setStyleFidity] = useState(0.5)
    const [controlReferenceImage, setControlReferenceImage] = useState(false)
    const [expandDialogFlag, setExpandDialogFlag] = useState(false)
    const [horizontalExpand, setHorizontalExpand] = useState("0")
    const [verticalExpand, setVerticalExpand] = useState("0")
    const [expandImage, setExpandImage] = useState("")
    const [expandMask, setExpandMask] = useState("")
    const [started, setStarted] = useState(false)
    const [loopMode, setLoopMode] = useState("repeat prompt")
    const [savedPrompts, setSavedPrompts] = useState([])
    const [viewImages, setViewImages] = useState([])
    const [imageName, setImageName] = useState("")
    const [upscaler, setUpscaler] = useState("real-esrgan")
    const [nsfwTab, setNSFWTab] = useState(false)
    const [watermark, setWatermark] = useState(true)
    const [aiWatermarkPosition, setAIWatermarkPosition] = useState("top left")
    const [aiWatermarkType, setAIWatermarkType] = useState("fan")
    const [aiWatermarkHue, setAIWatermarkHue] = useState(0)
    const [aiWatermarkSaturation, setAIWatermarkSaturation] = useState(0)
    const [aiWatermarkBrightness, setAIWatermarkBrightness] = useState(0)
    const [aiWatermarkInvert, setAIWatermarkInvert] = useState(false)
    const [aiWatermarkOpacity, setAIWatermarkOpacity] = useState(100)
    const [aiWatermarkMarginX, setAIWatermarkMarginX] = useState(10)
    const [aiWatermarkMarginY, setAIWatermarkMarginY] = useState(10)
    const [aiWatermarkScale, setAIWatermarkScale] = useState(0.7)

    return (
        <>  
            <NSFWTabContext.Provider value={{nsfwTab, setNSFWTab}}>
            <UpscalerContext.Provider value={{upscaler, setUpscaler}}>
            <AIWatermarkScaleContext.Provider value={{aiWatermarkScale, setAIWatermarkScale}}>
            <AIWatermarkMarginYContext.Provider value={{aiWatermarkMarginY, setAIWatermarkMarginY}}>
            <AIWatermarkMarginXContext.Provider value={{aiWatermarkMarginX, setAIWatermarkMarginX}}>
            <AIWatermarkOpacityContext.Provider value={{aiWatermarkOpacity, setAIWatermarkOpacity}}>
            <AIWatermarkInvertContext.Provider value={{aiWatermarkInvert, setAIWatermarkInvert}}>
            <AIWatermarkBrightnessContext.Provider value={{aiWatermarkBrightness, setAIWatermarkBrightness}}>
            <AIWatermarkSaturationContext.Provider value={{aiWatermarkSaturation, setAIWatermarkSaturation}}>
            <AIWatermarkHueContext.Provider value={{aiWatermarkHue, setAIWatermarkHue}}>
            <AIWatermarkTypeContext.Provider value={{aiWatermarkType, setAIWatermarkType}}>
            <AIWatermarkPositionContext.Provider value={{aiWatermarkPosition, setAIWatermarkPosition}}>
            <WatermarkContext.Provider value={{watermark, setWatermark}}>
            <ImageNameContext.Provider value={{imageName, setImageName}}>
            <ViewImagesContext.Provider value={{viewImages, setViewImages}}>
            <SavedPromptsContext.Provider value={{savedPrompts, setSavedPrompts}}>
            <LoopModeContext.Provider value={{loopMode, setLoopMode}}>
            <StartedContext.Provider value={{started, setStarted}}>
            <ExpandMaskContext.Provider value={{expandMask, setExpandMask}}>
            <ExpandImageContext.Provider value={{expandImage, setExpandImage}}>
            <VerticalExpandContext.Provider value={{verticalExpand, setVerticalExpand}}>
            <HorizontalExpandContext.Provider value={{horizontalExpand, setHorizontalExpand}}>
            <ExpandDialogFlagContext.Provider value={{expandDialogFlag, setExpandDialogFlag}}>
            <ControlReferenceImageContext.Provider value={{controlReferenceImage, setControlReferenceImage}}>
            <ImageContrastContext.Provider value={{imageContrast, setImageContrast}}>
            <ImageBrightnessContext.Provider value={{imageBrightness, setImageBrightness}}>
            <StyleFidelityContext.Provider value={{styleFidelity, setStyleFidity}}>
            <ControlInvertContext.Provider value={{controlInvert, setControlInvert}}>
            <ControlEndContext.Provider value={{controlEnd, setControlEnd}}>
            <ControlStartContext.Provider value={{controlStart, setControlStart}}>
            <ControlGuessModeContext.Provider value={{controlGuessMode, setControlGuessMode}}>
            <ControlScaleContext.Provider value={{controlScale, setControlScale}}>
            <ControlImageContext.Provider value={{controlImage, setControlImage}}>
            <ControlProcessorContext.Provider value={{controlProcessor, setControlProcessor}}>
            <SidebarTypeContext.Provider value={{sidebarType, setSidebarType}}>
            <PrecisionContext.Provider value={{precision, setPrecision}}>
            <MaskDataContext.Provider value={{maskData, setMaskData}}>
            <MaskImageContext.Provider value={{maskImage, setMaskImage}}>
            <DrawImageContext.Provider value={{drawImage, setDrawImage}}>
            <ThemeContext.Provider value={{theme, setTheme}}>
            <FormatContext.Provider value={{format, setFormat}}>
            <DeletionContext.Provider value={{deletion, setDeletion}}>
            <ImageInputContext.Provider value={{imageInput, setImageInput}}>
            <ProcessingContext.Provider value={{processing, setProcessing}}>
            <TabContext.Provider value={{tab, setTab}}>
            <VAENamesContext.Provider value={{vaeNames, setVAENames}}>
            <VAENameContext.Provider value={{vaeName, setVAEName}}>
            <PreviewImageContext.Provider value={{previewImage, setPreviewImage}}>
            <UpdateSavedContext.Provider value={{updateSaved, setUpdateSaved}}>
            <RenderImageContext.Provider value={{renderImage, setRenderImage}}>
            <ModelNamesContext.Provider value={{modelNames, setModelNames}}>
            <ModelNameContext.Provider value={{modelName, setModelName}}>
            <AmountContext.Provider value={{amount, setAmount}}>
            <ClipSkipContext.Provider value={{clipSkip, setClipSkip}}>
            <NegativePromptContext.Provider value={{negativePrompt, setNegativePrompt}}>
            <PromptContext.Provider value={{prompt, setPrompt}}>
            <InterrogatorNameContext.Provider value={{interrogatorName, setInterrogatorName}}>
            <InterrogateTextContext.Provider value={{interrogateText, setInterrogateText}}>
            <SamplerContext.Provider value={{sampler, setSampler}}>
            <SeedContext.Provider value={{seed, setSeed}}>
            <DenoiseContext.Provider value={{denoise, setDenoise}}>
            <SizeContext.Provider value={{size, setSize}}>
            <CFGContext.Provider value={{cfg, setCFG}}>
            <StepsContext.Provider value={{steps, setSteps}}>
            <SiteLightnessContext.Provider value={{siteLightness, setSiteLightness}}>
            <SiteSaturationContext.Provider value={{siteSaturation, setSiteSaturation}}>
            <SiteHueContext.Provider value={{siteHue, setSiteHue}}>
                {props.children}
            </SiteHueContext.Provider>
            </SiteSaturationContext.Provider>
            </SiteLightnessContext.Provider>
            </StepsContext.Provider>
            </CFGContext.Provider>
            </SizeContext.Provider>
            </DenoiseContext.Provider>
            </SeedContext.Provider>
            </SamplerContext.Provider>
            </InterrogateTextContext.Provider>
            </InterrogatorNameContext.Provider>
            </PromptContext.Provider>
            </NegativePromptContext.Provider>
            </ClipSkipContext.Provider>
            </AmountContext.Provider>
            </ModelNameContext.Provider>
            </ModelNamesContext.Provider>
            </RenderImageContext.Provider>
            </UpdateSavedContext.Provider>
            </PreviewImageContext.Provider>
            </VAENameContext.Provider>
            </VAENamesContext.Provider>
            </TabContext.Provider>
            </ProcessingContext.Provider>
            </ImageInputContext.Provider>
            </DeletionContext.Provider>
            </FormatContext.Provider>
            </ThemeContext.Provider>
            </DrawImageContext.Provider>
            </MaskImageContext.Provider>
            </MaskDataContext.Provider>
            </PrecisionContext.Provider>
            </SidebarTypeContext.Provider>
            </ControlProcessorContext.Provider>
            </ControlImageContext.Provider>
            </ControlScaleContext.Provider>
            </ControlGuessModeContext.Provider>
            </ControlStartContext.Provider>
            </ControlEndContext.Provider>
            </ControlInvertContext.Provider>
            </StyleFidelityContext.Provider>
            </ImageBrightnessContext.Provider>
            </ImageContrastContext.Provider>
            </ControlReferenceImageContext.Provider>
            </ExpandDialogFlagContext.Provider>
            </HorizontalExpandContext.Provider>
            </VerticalExpandContext.Provider>
            </ExpandImageContext.Provider>
            </ExpandMaskContext.Provider>
            </StartedContext.Provider>
            </LoopModeContext.Provider>
            </SavedPromptsContext.Provider>
            </ViewImagesContext.Provider>
            </ImageNameContext.Provider>
            </WatermarkContext.Provider>
            </AIWatermarkPositionContext.Provider>
            </AIWatermarkTypeContext.Provider>
            </AIWatermarkHueContext.Provider>
            </AIWatermarkSaturationContext.Provider>
            </AIWatermarkBrightnessContext.Provider>
            </AIWatermarkInvertContext.Provider>
            </AIWatermarkOpacityContext.Provider>
            </AIWatermarkMarginXContext.Provider>
            </AIWatermarkMarginYContext.Provider>
            </AIWatermarkScaleContext.Provider>
            </UpscalerContext.Provider>
            </NSFWTabContext.Provider>
        </>
    )
}

export default Context